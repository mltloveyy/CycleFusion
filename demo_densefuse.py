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
from fuser import WeightFuser
from losses import *
from utils import *

EPSILON = 1e-3
args = parser.parse_args()
current_time = time.strftime("%Y%m%d_%H%M%S")
args.output_dir = join_path(args.output_dir, current_time)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


def test(encoder, decoder, test_set: FingerPrint, epoch, write_result=False):
    logging.info("start testing...")
    encoder.eval()
    decoder.eval()
    fuser = WeightFuser(strategy=args.fuse_type)
    test_loss = 0.0

    testloader = DataLoader(test_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # run the model on the test set to predict
            f_tir = encoder(img_tir)
            f_oct = encoder(img_oct)
            f_fused = fuser.forward(f_tir, f_oct)
            tir_rec = decoder(f_tir)
            oct_rec = decoder(f_oct)
            img_fused = decoder(f_fused)

            restore_loss_value = args.restore_weight * (restore_loss(tir_rec, img_tir).item() + restore_loss(oct_rec, img_oct).item())
            total_loss_value = restore_loss_value
            test_loss += total_loss_value
            logging.debug(f"[Test loss] restore: {restore_loss_value:.5f}")

            # save testset results
            if write_result & (epoch % args.save_interval == 0):
                logging.debug(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                output = torch.cat((img_tir, tir_rec, img_oct, oct_rec, img_fused), dim=-1).cpu()
                save_tensor(output, join_path(save_dir, f"epoch_{epoch}_{test_set.filenames[i]}.jpg"))

    return test_loss / len(testloader)


def train(train_set: FingerPrint, test_set: FingerPrint):
    # Create output dir
    create_dirs(args.output_dir)

    # Init Tensorboard dir
    writer = SummaryWriter(join_path(args.output_dir, "summary_only_restore"))

    # Set logging
    logging.basicConfig(
        format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(join_path(args.output_dir, current_time + ".log")),
            logging.StreamHandler(),
        ],
    )

    # init models
    if args.network_type == "CDDFuse":
        encoder = CDDFuseEncoder()
        decoder = CDDFuseDecoder()
    else:
        encoder = DenseFuseEncoder()
        decoder = DenseFuseDecoder()
    load_model(args.pretrain_weight, encoder, "encoder")
    load_model(args.pretrain_weight, decoder, "decoder")
    encoder.to(device)
    decoder.to(device)

    loss_minimum = 10000.0

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):  # loop over the dataset multiple times
        logging.info(f"start training No.{epoch} epoch...")

        Loss_restore = []

        optimizer = Adam([{"params": encoder.parameters()}, {"params": decoder.parameters()}], lr=args.lr)

        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            # get the inputs
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # zero the encoder parameter gradients
            optimizer.zero_grad()

            # forward
            f_tir = encoder(img_tir)
            f_oct = encoder(img_oct)
            tir_rec = decoder(f_tir)
            oct_rec = decoder(f_oct)

            # backward
            restore_loss_value = args.restore_weight * (restore_loss(tir_rec, img_tir) + restore_loss(oct_rec, img_oct))
            restore_loss_value.backward()
            Loss_restore.append(restore_loss_value.item())
            writer.add_scalar("train_loss/restore", restore_loss_value.item(), epoch * len(trainloader) + i)
            logging.info(f"[Iter{i}] restore: {restore_loss_value.item():.5f}")

            # optimize
            optimizer.step()

        end_epoch = time.time()

        # logging.info cost
        logging.info(f"epoch: {epoch} time_taken: {end_epoch - start_epoch:.3f}")
        restore_mean = np.mean(np.array(Loss_restore))
        logging.info(f"[Train loss] restore: {restore_mean:.5f}")
        train_loss = restore_mean
        writer.add_scalar("train_loss", train_loss, epoch)
        logging.info(f"[Train loss] {train_loss :.5f} minimum: {loss_minimum :.5f}")
        if train_loss < loss_minimum:
            loss_minimum = train_loss
            best_epoch = epoch

        # Get loss on the test set
        test_loss = test(encoder, decoder, test_set, epoch, write_result=True)
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
            models_state = {
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
            }
            save_models(join_path(models_dir, f"epoch{epoch}.pth"), models_state)

        logging.info("========================================")

    return encoder, decoder


def eval(encoder, decoder, dir_tir, dir_oct, dir_output):
    cost = 0.0
    count = 0
    fuser = WeightFuser(strategy=args.fuse_type)
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
            f_tir = encoder(t_tir)
            f_oct = encoder(t_oct)
            f_fused = fuser.forward(f_tir, f_oct)
            img_fused = decoder(f_fused)
            cost += time.time() - start
            count += 1

            path_output = join_path(dir_output, img_name)
            save_tensor(img_fused.cpu().detach(), path_output)
    return cost / count


if __name__ == "__main__":
    args.fuse_type = "add"
    # Train
    logging.info(args)
    tir_data_path = join_path(args.data_dir, "tir")
    oct_data_path = join_path(args.data_dir, "oct")

    img_paths_tir = glob(join_path(tir_data_path, "*.bmp"))
    logging.info(f"Dataset size: {len(img_paths_tir)}")
    random.shuffle(img_paths_tir)
    test_paths = img_paths_tir[: args.test_num]
    train_paths = img_paths_tir[args.test_num :]
    train_set = FingerPrint(train_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=True, with_score=True)
    test_set = FingerPrint(test_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=False, with_score=True)
    logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    encoder, decoder = train(train_set, test_set)
    models_dir = join_path(args.output_dir, "models")
    create_dirs(models_dir)
    models_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
    }
    save_models(join_path(models_dir, f"final.pth"), models_state)

    # Eval
    model_path = "output/20240918_055413/models/final.pth"
    if not os.path.exists(model_path):
        print(f"model path:{model_path} is not exists")
        exit()
    data_dir = "experiment/raw_data/1208"
    dir_output = "experiment/results/1208_cddfuse_raw"
    create_dirs(dir_output)

    if args.network_type == "CDDFuse":
        encoder = CDDFuseEncoder()
        decoder = CDDFuseDecoder()
    else:
        encoder = DenseFuseEncoder()
        decoder = DenseFuseDecoder()

    load_model(model_path, encoder, "encoder")
    load_model(model_path, decoder, "decoder")
    encoder.to(device).eval()
    decoder.to(device).eval()

    dir_tir = data_dir + "/tir"
    dir_oct = data_dir + "/oct"
    cost = eval(encoder, decoder, dir_tir, dir_oct, dir_output)
    print(f"Average time: {cost}s")
