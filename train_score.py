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
from fusion import weight_fusion
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


def test(encoder, decoder, test_set: FingerPrint, epoch, write_result=False):
    logging.info("start testing...")
    encoder.eval()
    decoder.eval()
    test_loss = 0.0

    testloader = DataLoader(test_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)
            score_tir = data[2].to(device)
            score_oct = data[3].to(device)

            quality_loss_value, restore_loss_value, fuse_loss_value = 0, 0, 0

            # run the model on the test set to predict
            f_tir, score_tir_probs = encoder(img_tir)
            f_oct, score_oct_probs = encoder(img_oct)
            tir_rec = decoder(f_tir)
            oct_rec = decoder(f_oct)
            img_fused = decoder(f_tir, f_oct)
            _, score_fused_probs = encoder(img_fused)

            quality_loss_value = args.quality_weight * (quality_loss(score_tir_probs, score_tir).item() + quality_loss(score_oct_probs, score_oct).item())
            restore_loss_value = args.restore_weight * (restore_loss(tir_rec, img_tir).item() + restore_loss(oct_rec, img_oct).item())
            if args.with_better:
                union_mask = torch.logical_or((score_tir > EPSILON), (score_oct > EPSILON)).float()
                score_max = torch.maximum(score_tir, score_oct)
                union_mask = torch.clamp(1.2 * union_mask * score_max, 0, 1)
                fuse_loss_value = args.fuse_weight * (fuse_loss(score_fused_probs, union_mask).item() + fuse_loss(score_fused_probs, union_mask).item())

            total_loss_value = quality_loss_value + restore_loss_value + fuse_loss_value
            test_loss += total_loss_value
            logging.debug(f"[Test loss] quality: {quality_loss_value:.5f} restore: {restore_loss_value:.5f} fuse: {fuse_loss_value:.5f}")

            # save testset results
            if write_result & (epoch % args.save_interval == 0):
                logging.debug(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                output1 = torch.cat((img_tir, score_tir, img_oct, score_oct, img_fused), dim=-1)
                output2 = torch.cat((tir_rec, score_tir_probs, oct_rec, score_oct_probs, score_fused_probs), dim=-1)
                output = torch.cat((output1, output2), dim=-2).cpu()
                save_tensor(output, join_path(save_dir, f"epoch_{epoch}_{test_set.filenames[i]}.jpg"))

    return test_loss / len(testloader)


def test_only_restore(encoder, decoder, test_set: FingerPrint, epoch, write_result=False):
    logging.info("start testing...")
    encoder.eval()
    decoder.eval()
    test_loss = 0.0

    testloader = DataLoader(test_set, batch_size=1, shuffle=False)
    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # run the model on the test set to predict
            f_tir, _ = encoder(img_tir)
            f_oct, _ = encoder(img_oct)
            f_fused = weight_fusion(f_tir, f_oct, strategy_type="add")
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
    # init models
    if args.network_type == "CDDFuse":
        encoder = CDDFuseEncoder()
        decoder = CDDFuseDecoder()
    else:
        encoder = DenseFuseEncoder()
        decoder = DenseFuseDecoder()
    load_model(args.pretrain_weight + "_enc.pth", encoder)
    load_model(args.pretrain_weight + "_dec.pth", decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    loss_minimum = 10000.0

    trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    for epoch in range(args.epochs):
        logging.info(f"start training No.{epoch} epoch...")

        Loss_quality, Loss_restore, Loss_fuse = [], [], []

        optimizer_en = Adam(encoder.parameters(), lr=args.lr)
        optimizer_de = Adam(decoder.parameters(), lr=args.lr)

        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            # get the inputs
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)
            score_tir = data[2].to(device)
            score_oct = data[3].to(device)

            # zero the encoder parameter gradients
            optimizer_en.zero_grad()

            # encoder forward
            _, score_tir_probs = encoder(img_tir)
            _, score_oct_probs = encoder(img_oct)

            # encoder backward
            quality_loss_value = args.quality_weight * (quality_loss(score_tir_probs, score_tir) + quality_loss(score_oct_probs, score_oct))
            encoder_loss = quality_loss_value
            encoder_loss.backward()
            Loss_quality.append(quality_loss_value.item())
            writer.add_scalar("train_loss/quality", quality_loss_value.item(), epoch * len(trainloader) + i)

            # optimize encoder
            optimizer_en.step()

            if i % args.critic == 0:
                # froze encoder & decoder
                for param in encoder.parameters():
                    param.requires_grad = False

                # zero the decoder parameter gradients
                optimizer_de.zero_grad()

                # decoder forward
                f_tir, score_tir_probs = encoder(img_tir)
                f_oct, score_oct_probs = encoder(img_oct)
                tir_rec = decoder(f_tir)
                oct_rec = decoder(f_oct)
                img_fused = decoder(f_tir, f_oct)
                if args.with_better:
                    _, score_fused_probs = encoder(img_fused)
                    union_mask = torch.logical_or((score_tir > EPSILON), (score_oct > EPSILON)).float()
                    score_max = torch.maximum(score_tir, score_oct)
                    union_mask = torch.clamp(1.2 * union_mask * score_max, 0, 1)

                    # decoder backward
                    restore_loss_value = args.restore_weight * (restore_loss(tir_rec, img_tir) + restore_loss(oct_rec, img_oct))
                    fuse_loss_value = args.fuse_weight * (fuse_loss(score_fused_probs, union_mask) + fuse_loss(score_fused_probs, union_mask))
                    decoder_loss = restore_loss_value + fuse_loss_value
                    decoder_loss.backward()
                    Loss_restore.append(restore_loss_value.item())
                    Loss_fuse.append(fuse_loss_value.item())
                    writer.add_scalar("train_loss/restore", restore_loss_value.item(), epoch * len(trainloader) + i)
                    writer.add_scalar("train_loss/fuse", fuse_loss_value.item(), epoch * len(trainloader) + i)
                    logging.info(
                        f"[Iter{i}] quality: {quality_loss_value.item():.5f} restore: {restore_loss_value.item():.5f} fuse: {fuse_loss_value.item():.5f}"
                    )
                else:
                    # decoder backward
                    restore_loss_value = args.restore_weight * (restore_loss(tir_rec, img_tir) + restore_loss(oct_rec, img_oct))
                    decoder_loss = restore_loss_value
                    decoder_loss.backward()
                    Loss_restore.append(restore_loss_value.item())
                    writer.add_scalar("train_loss/restore", restore_loss_value.item(), epoch * len(trainloader) + i)
                    logging.info(f"[Iter{i}] quality: {quality_loss_value.item():.5f} restore: {restore_loss_value.item():.5f}")

                # optimize decoder
                optimizer_de.step()

                # unfroze encoder & decoder
                for param in encoder.parameters():
                    param.requires_grad = True

        end_epoch = time.time()

        # logging.info cost
        logging.info(f"epoch: {epoch} time_taken: {end_epoch - start_epoch:.3f}")
        quality_mean = np.mean(np.array(Loss_quality))
        restore_mean = np.mean(np.array(Loss_restore))
        if args.with_better:
            fuse_mean = np.mean(np.array(Loss_fuse))
            logging.info(f"[Train loss] quality: {quality_mean:.5f} restore: {restore_mean:.5f} fuse: {fuse_mean:.5f}")
            train_loss = quality_mean + restore_mean + fuse_mean
        else:
            logging.info(f"[Train loss] quality: {quality_mean:.5f} restore: {restore_mean:.5f}")
            train_loss = quality_mean + restore_mean
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
            save_model(join_path(models_dir, f"epoch{epoch}_enc.pth"), encoder)
            save_model(join_path(models_dir, f"epoch{epoch}_dec.pth"), decoder)

        logging.info("========================================")

    return encoder, decoder


def train_only_restore(train_set: FingerPrint, test_set: FingerPrint):
    # init models
    if args.network_type == "CDDFuse":
        encoder = CDDFuseEncoder()
        decoder = CDDFuseDecoder()
    else:
        encoder = DenseFuseEncoder()
        decoder = DenseFuseDecoder()
    load_model(args.pretrain_weight + "_enc.pth", encoder)
    load_model(args.pretrain_weight + "_dec.pth", decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

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
            f_tir, _ = encoder(img_tir)
            f_oct, _ = encoder(img_oct)
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
        test_loss = test_only_restore(encoder, decoder, test_set, epoch, write_result=True)
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

    return encoder, decoder


if __name__ == "__main__":
    logging.info(args)
    # Fetch dataset
    tir_data_path = join_path(args.data_dir, "tir")
    oct_data_path = join_path(args.data_dir, "oct")

    # Split dataset
    img_paths_tir = glob(join_path(tir_data_path, "*.bmp"))
    logging.info(f"Dataset size: {len(img_paths_tir)}")
    random.shuffle(img_paths_tir)
    test_paths = img_paths_tir[: args.test_num]
    train_paths = img_paths_tir[args.test_num :]
    train_set = FingerPrint(train_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=True, with_score=True)
    test_set = FingerPrint(test_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=False, with_score=True)
    logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")

    # Train models
    if args.with_quality:
        encoder, decoder = train(train_set, test_set)
        models_dir = join_path(args.output_dir, "models")
        create_dirs(models_dir)
        save_model(join_path(models_dir, f"final_enc.pth"), encoder)
        save_model(join_path(models_dir, f"final_dec.pth"), decoder)
    else:
        encoder, decoder = train_only_restore(train_set, test_set)
        models_dir = join_path(args.output_dir, "models")
        create_dirs(models_dir)
        save_model(join_path(models_dir, f"final_enc.pth"), encoder)
        save_model(join_path(models_dir, f"final_dec.pth"), decoder)
