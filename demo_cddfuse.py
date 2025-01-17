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
from losses import *
from networks import QualityFuser, cddfuse_offical
from utils import *

EPSILON = 1e-3
args = parser.parse_args()
current_time = time.strftime("%Y%m%d_%H%M%S")
args.output_dir = join_path(args.output_dir, "cddfuse_" + current_time)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


def test(model: QualityFuser, dataset: FingerPrint, epoch, write_result=False):
    logging.info("start testing...")
    model.to("eval")
    test_loss = 0.0

    testloader = DataLoader(dataset, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # run the model on the test set to predict
            model.encode(img_tir, img_oct)
            tir_rec, oct_rec = model.decode()
            img_fused, _ = model.fuse()

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
    writer = SummaryWriter(join_path(args.output_dir, "summary_cddfuse"))

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
    model = cddfuse_offical
    model.load(args.pretrain_weight)
    model.to("train", device)

    loss_minimum = 10000.0

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        logging.info(f"start training No.{epoch} epoch...")

        Loss = []

        optimizer = Adam([{"params": model.encoder.parameters()}, {"params": model.decoder.parameters()}], lr=args.lr)
        optimizer_fuse = Adam(model.fuser.parameters(), lr=args.lr)

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
                model.encode(img_tir, img_oct)
                tir_rec, oct_rec = model.decode()

                # backward
                loss_tir = 5 * ssim_loss(img_tir, tir_rec) + mse_loss(img_tir, tir_rec)
                loss_oct = 5 * ssim_loss(img_oct, oct_rec) + mse_loss(img_oct, oct_rec)

                cc_loss_base = cc(model.f1_base, model.f2_base)
                cc_loss_detail = cc(model.f1_detail, model.f2_detail)
                loss_decomp = (cc_loss_detail) ** 2 / (1.01 + cc_loss_base)

                loss = loss_tir + loss_oct + 2 * loss_decomp
                loss.backward()

                # optimize
                optimizer.step()
            else:
                # forward
                model.encode(img_tir, img_oct)
                img_fused, _ = model.fuse()

                # backward
                cc_loss_base = cc(model.f1_base, model.f2_base)
                cc_loss_detail = cc(model.f1_detail, model.f2_detail)
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
        test_loss = test(model, test_set, epoch, write_result=True)
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
            model.save(join_path(models_dir, f"epoch{epoch}.pth"))

        logging.info("========================================")

    return model


if __name__ == "__main__":
    args.network_type = "cddfuse"
    args.fuse_type = "cddfuse"
    args.with_quality = False
    args.with_revaluate = False
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

    model = train(train_set, test_set)
    models_dir = join_path(args.output_dir, "models")
    create_dirs(models_dir)
    model.save(join_path(models_dir, "final.pth"))
