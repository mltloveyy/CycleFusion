import logging
import random
import time
from glob import glob

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from args import parser
from data_process import FingerPrint
from decoder import CDDFuseDecoder, DenseFuseDecoder
from encoder import CDDFuseEncoder, DenseFuseEncoder
from fusion import CDDFuseFuser, weight_fusion
from losses import *
from utils import *

EPSILON = 1e-3

# Create output dir
args = parser.parse_args()
current_time = time.strftime("%Y%m%d_%H%M%S")
args.output_dir = join_path(args.output_dir, current_time)
create_dirs(args.output_dir)

# Init Tensorboard dir
writer = SummaryWriter(join_path(args.output_dir, "summary"))

# Set logging
logging.basicConfig(
    format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(join_path(args.output_dir, current_time + ".log")),
        logging.StreamHandler(),
    ],
)

# Set device
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


def test(encoder, decoder, fuser, test_set: FingerPrint, epoch, write_result=False):
    logging.info("start testing...")
    encoder.eval()
    decoder.eval()
    fuser.eval()
    test_loss = 0.0

    testloader = DataLoader(test_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # run the model on the test set to predict
            f_tir_base, f_tir_detail = encoder(img_tir, with_score=False)
            f_oct_base, f_oct_detail = encoder(img_oct, with_score=False)
            f_fuse_base, f_fuse_detail = fuser(f_tir_base + f_oct_base, f_tir_detail + f_oct_detail)
            img_fused = decoder(f_fuse_base, f_fuse_detail)

            # save testset results
            if write_result & (epoch % args.save_interval == 0):
                logging.debug(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                output = torch.cat((img_tir, img_oct, img_fused), dim=-1).cpu()
                save_tensor(output, join_path(save_dir, f"epoch_{epoch}_{test_set.filenames[i]}.jpg"))

    return test_loss / len(testloader)


def train(train_set: FingerPrint, test_set: FingerPrint):
    # init models
    args.network_type = "CDDFuse"
    encoder = CDDFuseEncoder()
    decoder = CDDFuseDecoder()
    fuser = CDDFuseFuser()
    load_model(args.pretrain_weight + "_enc.pth", encoder)
    load_model(args.pretrain_weight + "_dec.pth", decoder)
    load_model(args.pretrain_weight + "_fuse.pth", fuser)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    fuser = fuser.to(device)

    loss_minimum = 10000.0

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        logging.info(f"start training No.{epoch} epoch...")

        Loss = []

        optimizer = Adam([{"params": encoder.parameters()}, {"params": decoder.parameters()}], lr=args.lr)
        optimizer_fuse = Adam(fuser.parameters(), lr=args.lr)

        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            # get the inputs
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
            optimizer_fuse.zero_grad()

            if epoch < 40:
                # forward
                f_tir_base, f_tir_detail = encoder(img_tir, with_score=False)
                f_oct_base, f_oct_detail = encoder(img_oct, with_score=False)
                tir_rec = decoder(f_tir_base, f_tir_detail)
                oct_rec = decoder(f_oct_base, f_oct_detail)

                # backward
                loss_tir = 5 * ssim_loss(img_tir, tir_rec) + mse_loss(img_tir, tir_rec)
                loss_oct = 5 * ssim_loss(img_oct, oct_rec) + mse_loss(img_oct, oct_rec)

                cc_loss_base = cc(f_tir_base, f_oct_base)
                cc_loss_detail = cc(f_tir_detail, f_oct_detail)
                loss_decomp = (cc_loss_detail) ** 2 / (1.01 + cc_loss_base)

                loss = loss_tir + loss_oct + 2 * loss_decomp
                loss.backward()

                # optimize
                optimizer.step()
            else:
                # forward
                f_tir_base, f_tir_detail = encoder(img_tir, with_score=False)
                f_oct_base, f_oct_detail = encoder(img_oct, with_score=False)
                f_fuse_base, f_fuse_detail = fuser(f_tir_base + f_oct_base, f_tir_detail + f_oct_detail)
                img_fused = decoder(f_fuse_base, f_fuse_detail)

                # backward
                cc_loss_base = cc(f_tir_base, f_oct_base)
                cc_loss_detail = cc(f_tir_detail, f_oct_detail)
                loss_decomp = (cc_loss_detail) ** 2 / (1.01 + cc_loss_base)

                fusion_loss, _, _ = cddfuse_loss(img_tir, img_oct, img_fused)

                loss = fusion_loss + 2 * loss_decomp
                loss.backward()

                # optimize
                optimizer.step()
                optimizer_fuse.step()

            Loss.append(loss.item())
            writer.add_scalar("train_loss/loss", loss.item(), epoch * len(trainloader) + i)
            logging.info(f"[Iter{i}] loss: {loss.item():.5f}")

        end_epoch = time.time()

        # logging.info cost
        logging.info(f"epoch: {epoch} time_taken: {end_epoch - start_epoch:.3f}")
        loss_mean = np.mean(np.array(Loss))
        logging.info(f"[Train loss] loss: {loss_mean:.5f}")
        train_loss = loss_mean
        writer.add_scalar("train_loss", train_loss, epoch)
        logging.info(f"[Train loss] {train_loss :.5f} minimum: {loss_minimum :.5f}")
        if train_loss < loss_minimum:
            loss_minimum = train_loss
            best_epoch = epoch

        # Get loss on the test set
        test_loss = test(encoder, decoder, fuser, test_set, epoch, write_result=True)
        writer.add_scalar("test_loss", test_loss, epoch)
        logging.info(f"[Test loss] {test_loss:.5f}")

        # Condition for reduce lr and early stopping
        if epoch - best_epoch > args.patience:
            args.lr /= 10
            best_epoch = epoch
            logging.info(f"train loss does not drop, the learning rate will be reduced to {args.lr}")
            if args.lr < 1e-5:
                logging.info(f"Early stopping at epoch {epoch}")
                break

        # Save checkpoints
        if (epoch > 10) & (epoch % args.save_interval == 0):
            models_dir = join_path(args.output_dir, "models")
            create_dirs(models_dir)
            save_model(join_path(models_dir, f"epoch{epoch}_enc.pth"), encoder)
            save_model(join_path(models_dir, f"epoch{epoch}_dec.pth"), decoder)

        logging.info("========================================")

    return encoder, decoder, fuser


def eval_fuse(encoder, decoder, fuser, dir_tir, dir_oct, dir_output):
    cost = 0.0
    count = 0
    with torch.no_grad():
        for img_name in os.listdir(dir_tir):
            if not img_name.endswith(".bmp"):
                continue

            path_tir = join_path(dir_tir, img_name)
            path_oct = join_path(dir_oct, img_name)

            img_tir = read_image(path_tir).transpose(2, 0, 1)
            img_oct = read_image(path_oct).transpose(2, 0, 1)
            t_tir = torch.unsqueeze(torch.from_numpy(img_tir) / 255.0, dim=0).to(device)
            t_oct = torch.unsqueeze(torch.from_numpy(img_oct) / 255.0, dim=0).to(device)

            start = time.time()
            f_tir_base, f_tir_detail = encoder(t_tir, with_score=False)
            f_oct_base, f_oct_detail = encoder(t_oct, with_score=False)
            f_fuse_base, f_fuse_detail = fuser(f_tir_base + f_oct_base, f_tir_detail + f_oct_detail)
            img_fused = decoder(f_fuse_base, f_fuse_detail)
            cost += time.time() - start
            count += 1

            path_output = join_path(dir_output, img_name)
            save_tensor(img_fused.cpu().detach(), path_output)
    return cost / count


if __name__ == "__main__":
    # Train

    # logging.info(args)
    # tir_data_path = join_path(args.data_dir, "tir")
    # oct_data_path = join_path(args.data_dir, "oct")

    # img_paths_tir = glob(join_path(tir_data_path, "*.bmp"))
    # logging.info(f"Dataset size: {len(img_paths_tir)}")
    # random.shuffle(img_paths_tir)
    # test_paths = img_paths_tir[: args.test_num]
    # train_paths = img_paths_tir[args.test_num :]
    # train_set = FingerPrint(train_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=True, with_score=True)
    # test_set = FingerPrint(test_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=False, with_score=True)
    # logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    # encoder, decoder, fuser = train(train_set, test_set)
    # models_dir = join_path(args.output_dir, "models")
    # create_dirs(models_dir)
    # save_model(join_path(models_dir, f"final_enc.pth"), encoder)
    # save_model(join_path(models_dir, f"final_dec.pth"), decoder)
    # save_model(join_path(models_dir, f"final_fuse.pth"), fuser)

    # Eval
    model_base = "output/20240918_055413/models/final"
    data_dir = "experiment/raw_data/1208"
    dir_output = "experiment/results/1208_cddfuse_raw"
    create_dirs(dir_output)

    encoder = CDDFuseEncoder()
    load_model(model_base + "_enc.pth", encoder)
    encoder.to(device)
    encoder.eval()

    decoder = CDDFuseDecoder()
    load_model(model_base + "_dec.pth", decoder)
    decoder.to(device)
    decoder.eval()

    fuser = CDDFuseFuser()
    load_model(model_base + "_fuse.pth", fuser)
    fuser.to(device)
    fuser.eval()

    dir_tir = data_dir + "/tir"
    dir_oct = data_dir + "/oct"
    cost = eval_fuse(encoder, decoder, fuser, dir_tir, dir_oct, dir_output)
    print(f"Average time: {cost}s")
