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
from networks import DeformedFuser
from quality import calc_quality_torch
from utils import *

EPSILON = 1e-3
args = parser.parse_args()
current_time = time.strftime("%Y%m%d_%H%M%S")
args.output_dir = join_path(args.output_dir, "deform_" + current_time)
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")


def test(model: DeformedFuser, test_set: FingerPrint, epoch, write_result=False):
    logging.info("start testing...")
    model.to("eval")
    test_loss = 0.0

    testloader = DataLoader(test_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            restore_loss_value, ncc_loss_value, flow_loss_value = 0, 0, 0

            # run the model on the test set to predict
            model.encode(img_tir, img_oct)
            rec_tir, rec_oct = model.decode()
            reg_oct, trans_oct, flow = model.deform(img_oct)

            restore_loss_value = args.restore_weight * (restore_loss(rec_tir, img_tir) + restore_loss(rec_oct, img_oct)).item()
            ncc_loss_value = ncc_loss(reg_oct, img_tir).item() + ncc_loss(trans_oct, img_tir).item()
            flow_loss_value = grad_loss(flow).item()

            total_loss_value = restore_loss_value + ncc_loss_value + flow_loss_value
            test_loss += total_loss_value
            logging.debug(f"[Test loss] restore: {restore_loss_value:.5f} ncc: {ncc_loss_value:.5f} flow: {flow_loss_value:.5f}")

            # save testset results
            if write_result & (epoch % args.save_interval == 0):
                logging.debug(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                rec_cmp = torch.cat((img_tir, rec_tir, img_oct, rec_oct), dim=-1).cpu()
                deform_cmp = torch.cat((img_tir, img_oct, trans_oct, reg_oct), dim=-1).cpu()
                output = torch.cat((rec_cmp, deform_cmp), dim=-2).cpu()
                save_tensor(output, join_path(save_dir, f"epoch_{epoch}_{test_set.filenames[i]}.jpg"))

    return test_loss / len(testloader)


def train(train_set: FingerPrint, test_set: FingerPrint):
    # Create output dir
    create_dirs(args.output_dir)

    # Init Tensorboard dir
    writer = SummaryWriter(join_path(args.output_dir, "summary_deform"))

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
    model = DeformedFuser(
        network_type=args.network_type,
        fuse_type=None,
        path=args.pretrain_weight,
        pred_affine_mat=False,
        img_size=(args.image_size, args.image_size),
    )
    model.to("train", device)

    loss_minimum = 10000.0

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        logging.info(f"start training No.{epoch} epoch...")

        Loss_restore, Loss_ncc, Loss_flow = [], [], []

        optimizer1 = Adam([{"params": model.encoder.parameters()}, {"params": model.decoder.parameters()}], lr=args.lr)
        optimizer2 = Adam(model.deformer.parameters(), lr=args.lr)

        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            # get the inputs
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # zero the restorer parameter gradients
            optimizer1.zero_grad()

            # restorer forward
            model.encode(img_tir, img_oct)
            rec_tir, rec_oct = model.decode()

            # restorer backward
            restore_loss_value = args.restore_weight * (restore_loss(rec_tir, img_tir) + restore_loss(rec_oct, img_oct))
            loss1 = restore_loss_value
            loss1.backward()
            Loss_restore.append(restore_loss_value.item())
            writer.add_scalar("train_loss/restore", restore_loss_value.item(), epoch * len(trainloader) + i)

            # optimize restorer
            optimizer1.step()

            if i % args.critic == 0:
                # froze restorer
                for param in model.encoder.parameters():
                    param.requires_grad = False
                for param in model.decoder.parameters():
                    param.requires_grad = False

                # zero the deformer parameter gradients
                optimizer2.zero_grad()

                # deformer forward
                reg_oct, trans_oct, flow = model.deform(img_oct)

                # deformer backward
                ncc_loss_value = ncc_loss(reg_oct, img_tir) + ncc_loss(trans_oct, img_tir)
                flow_loss_value = grad_loss(flow)
                loss2 = ncc_loss_value + flow_loss_value
                loss2.backward()
                Loss_ncc.append(ncc_loss_value.item())
                Loss_flow.append(flow_loss_value.item())
                writer.add_scalar("train_loss/ncc", ncc_loss_value.item(), epoch * len(trainloader) + i)
                writer.add_scalar("train_loss/flow", flow_loss_value.item(), epoch * len(trainloader) + i)
                logging.info(f"[Iter{i}] restore: {restore_loss_value.item():.5f} ncc: {ncc_loss_value.item():.5f} flow: {flow_loss_value.item():.5f}")

                # optimize deformer
                optimizer2.step()

                # unfroze restorer
                for param in model.encoder.parameters():
                    param.requires_grad = True
                for param in model.decoder.parameters():
                    param.requires_grad = True

        end_epoch = time.time()

        # logging.info cost
        logging.info(f"epoch: {epoch} time_taken: {end_epoch - start_epoch:.3f}")
        restore_mean = np.mean(np.array(Loss_restore))
        ncc_mean = np.mean(np.array(Loss_ncc))
        flow_mean = np.mean(np.array(Loss_flow))
        logging.info(f"[Train loss] restore: {restore_mean:.5f} ncc: {ncc_mean:.5f} flow: {flow_mean:.5f}")
        train_loss = restore_mean + ncc_mean + flow_mean
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
