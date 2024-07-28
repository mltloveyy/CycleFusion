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

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-4

# Create output dir
args = parser.parse_args()
current_time = time.strftime("%Y%m%d_%H%M%S")
args.output_dir = join_path(args.output_dir, current_time)
create_dirs(args.output_dir)

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


def test(encoder, decoder, testloader, epoch, write_result=False):
    logging.info("start testing...")
    encoder.eval()
    if decoder is not None:
        decoder.eval()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)
            score_tir = data[2].to(device)
            score_oct = data[3].to(device)

            quality_loss_value, ssim_loss_value, restorm_loss_value, pixel_loss_value, better_fusion_loss_value = 0, 0, 0, 0, 0

            # run the model on the test set to predict
            f_tir, score_tir_probs = encoder(img_tir)
            f_oct, score_oct_probs = encoder(img_oct)
            quality_loss_value = args.quality_weight * (mae_loss(score_tir_probs, score_tir).item() + mae_loss(score_oct_probs, score_oct).item())
            ssim_loss_value = args.ssim_weight * (ssim_loss(score_tir_probs, score_tir) + ssim_loss(score_oct_probs, score_oct))
            if decoder is not None:
                f_fused = weight_fusion(f_tir, f_oct, score_tir_probs, score_oct_probs, strategy_type="power")
                tir_rec = decoder(f_tir)
                oct_rec = decoder(f_oct)
                img_fused = decoder(f_fused)
                _, score_fused_probs = encoder(img_fused)
                union_mask = torch.logical_or((score_tir > EPSILON), (score_oct > EPSILON)).float()
                # score_max = torch.maximum(score_tir, score_oct)
                # union_mask = torch.clamp(1.2 * score_max, 0, 1)

                restorm_loss_value = args.ssim_weight * (ssim_loss(tir_rec, img_tir).item() + ssim_loss(oct_rec, img_oct).item())
                pixel_loss_value = args.pixel_weight * (mse_loss(tir_rec, img_tir).item() + mse_loss(oct_rec, img_oct).item())
                # better_fusion_loss_value = args.fuse_weight * (mse_loss(score_fused_probs, union_mask).item() + ssim_loss(score_fused_probs, union_mask).item())
                better_fusion_loss_value = args.fuse_weight * mse_loss(score_fused_probs, union_mask).item()

            total_loss_value = quality_loss_value + restorm_loss_value + ssim_loss_value + pixel_loss_value + better_fusion_loss_value
            test_loss += total_loss_value
            logging.debug(
                f"[Test loss] quality: {quality_loss_value:.5f} ssim: {ssim_loss_value:.5f} restorm: {restorm_loss_value:.5f} pixel: {pixel_loss_value:.5f} fusion: {better_fusion_loss_value:.5f}"
            )

            # save testset results
            if write_result & (epoch % args.save_interval == 0):
                logging.info(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                score_cmp = torch.cat((score_tir, score_tir_probs, score_oct, score_oct_probs), dim=-1).cpu()
                save_tensor(score_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_score_cmp.jpg"))
                if decoder is not None:
                    imgs_cmp = torch.cat((img_tir, tir_rec, img_oct, oct_rec), dim=-1).cpu()
                    fuse_cmp = torch.cat((img_tir, img_oct, img_fused, score_fused_probs), dim=-1).cpu()
                    save_tensor(imgs_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_img_cmp.jpg"))
                    save_tensor(fuse_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_fuse_cmp.jpg"))

    return test_loss / len(testloader)


def train(trainloader, testloader):
    # init models
    encoder = CDDFuseEncoder()
    decoder = CDDFuseDecoder()
    load_model(args.pretrain_weight + "_enc.pth", encoder)
    load_model(args.pretrain_weight + "_dec.pth", decoder)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    loss_minimum = 10000.0

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        logging.info(f"start training No.{epoch} epoch...")

        Loss_quality, Loss_ssim, Loss_restorm, Loss_pixel, Loss_fusion = [], [], [], [], []

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
            quality_loss_value = args.quality_weight * (mae_loss(score_tir_probs, score_tir) + mae_loss(score_oct_probs, score_oct))
            ssim_loss_value = args.ssim_weight * (ssim_loss(score_tir_probs, score_tir) + ssim_loss(score_oct_probs, score_oct))
            encoder_loss = quality_loss_value + ssim_loss_value
            encoder_loss.backward()
            Loss_quality.append(quality_loss_value.item())
            Loss_ssim.append(ssim_loss_value.item())
            writer.add_scalar("train_loss/quality", quality_loss_value.item(), epoch * len(trainloader) + i)
            writer.add_scalar("train_loss/ssim", ssim_loss_value.item(), epoch * len(trainloader) + i)

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
                f_fused = weight_fusion(f_tir, f_oct, score_tir_probs, score_oct_probs, strategy_type="power")
                tir_rec = decoder(f_tir)
                oct_rec = decoder(f_oct)
                img_fused = decoder(f_fused)
                _, score_fused_probs = encoder(img_fused)
                union_mask = torch.logical_or((score_tir > EPSILON), (score_oct > EPSILON)).float()
                # score_max = torch.maximum(score_tir, score_oct)
                # union_mask = torch.clamp(1.1 * score_max, 0, 1)

                # decoder backward
                restorm_loss_value = args.ssim_weight * (ssim_loss(tir_rec, img_tir) + ssim_loss(oct_rec, img_oct))
                pixel_loss_value = args.pixel_weight * (mse_loss(tir_rec, img_tir) + mse_loss(oct_rec, img_oct))
                # better_fusion_loss_value = args.fuse_weight * (mse_loss(score_fused_probs, union_mask) + ssim_loss(score_fused_probs, union_mask))
                better_fusion_loss_value = args.fuse_weight * mse_loss(score_fused_probs, union_mask)
                decoder_loss = restorm_loss_value + pixel_loss_value + better_fusion_loss_value
                decoder_loss.backward()
                Loss_restorm.append(restorm_loss_value.item())
                Loss_pixel.append(pixel_loss_value.item())
                Loss_fusion.append(better_fusion_loss_value.item())
                writer.add_scalar("train_loss/restorm", restorm_loss_value.item(), epoch * len(trainloader) + i)
                writer.add_scalar("train_loss/pixel", pixel_loss_value.item(), epoch * len(trainloader) + i)
                writer.add_scalar("train_loss/fusion", better_fusion_loss_value.item(), epoch * len(trainloader) + i)
                logging.info(
                    f"[Iter{i}] quality: {quality_loss_value.item():.5f} ssim: {ssim_loss_value.item():.5f} restorm: {restorm_loss_value.item():.5f} pixel: {pixel_loss_value.item():.5f} fusion: {better_fusion_loss_value.item():.5f}"
                )

                # optimize decoder
                optimizer_de.step()

                # unfroze encoder & decoder
                for param in encoder.parameters():
                    param.requires_grad = True

        end_epoch = time.time()

        # logging.info cost
        logging.info(f"epoch: {epoch} time_taken: {end_epoch - start_epoch:.3f}")
        quality_mean = np.mean(np.array(Loss_quality))
        ssim_mean = np.mean(np.array(Loss_ssim))
        restorm_mean = np.mean(np.array(Loss_restorm))
        pixel_mean = np.mean(np.array(Loss_pixel))
        fusion_mean = np.mean(np.array(Loss_fusion))
        logging.info(
            f"[Train loss] quality: {quality_mean:.5f} ssim: {ssim_mean:.5f} restorm: {restorm_mean:.5f} pixel: {pixel_mean:.5f} fusion: {fusion_mean:.5f}"
        )
        train_loss = quality_mean + ssim_mean + restorm_mean + pixel_mean + fusion_mean
        writer.add_scalar("train_loss", train_loss, epoch)
        logging.info(f"[Train loss] {train_loss :.5f} minimum: {loss_minimum :.5f}")
        if train_loss < loss_minimum:
            loss_minimum = train_loss
            best_epoch = epoch

        # Get loss on the test set
        test_loss = test(encoder, decoder, testloader, epoch, write_result=True)
        writer.add_scalar("test_loss", test_loss, epoch)
        logging.info(f"[Test loss] {test_loss:.5f}")

        # Condition for reduce lr and early stopping
        if epoch - best_epoch > args.patience:
            args.lr /= 10
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


def train_step(trainloader, testloader):
    # init models
    encoder = CDDFuseEncoder()
    decoder = None

    if args.training_encoder:
        load_model(args.pretrain_weight + "_enc.pth", encoder)
        encoder = encoder.to(device)
    else:
        # froze encoder
        encoder_path = args.pretrain_weight + "_enc.pth"
        if not os.path.exists(encoder_path):
            raise ValueError(f"encoder isn't existed: {encoder_path}")
        load_model(encoder_path, encoder)
        encoder = encoder.to(device)
        for param in encoder.parameters():
            param.requires_grad = False
        decoder = CDDFuseDecoder()
        load_model(args.pretrain_weight + "_dec.pth", decoder)
        decoder = decoder.to(device)

    loss_minimum = 10000.0

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        logging.info(f"start training No.{epoch} epoch...")

        Loss_quality, Loss_ssim, Loss_restorm, Loss_pixel, Loss_fusion = [], [], [], [], []

        if args.training_encoder:
            optimizer = Adam(encoder.parameters(), lr=args.lr)
        else:
            optimizer = Adam(decoder.parameters(), lr=args.lr)

        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            # get the inputs
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)
            score_tir = data[2].to(device)
            score_oct = data[3].to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # encoder forward
            f_tir, score_tir_probs = encoder(img_tir)
            f_oct, score_oct_probs = encoder(img_oct)

            if decoder is None:
                # encoder backward
                quality_loss_value = args.quality_weight * (mae_loss(score_tir_probs, score_tir) + mae_loss(score_oct_probs, score_oct))
                ssim_loss_value = args.ssim_weight * (ssim_loss(score_tir_probs, score_tir) + ssim_loss(score_oct_probs, score_oct))
                encoder_loss = quality_loss_value + ssim_loss_value
                encoder_loss.backward()
                Loss_quality.append(quality_loss_value.item())
                Loss_ssim.append(ssim_loss_value.item())

                # optimize encoder
                optimizer.step()

                logging.info(f"[Iter{i}] quality: {quality_loss_value.item():.5f} ssim: {ssim_loss_value.item():.5f}")
                writer.add_scalar("train_loss/quality", quality_loss_value.item(), epoch * len(trainloader) + i)
                writer.add_scalar("train_loss/ssim", ssim_loss_value.item(), epoch * len(trainloader) + i)
            else:
                # decoder forward
                f_fused = weight_fusion(f_tir, f_oct, score_tir_probs, score_oct_probs, strategy_type="power")
                tir_rec = decoder(f_tir)
                oct_rec = decoder(f_oct)
                img_fused = decoder(f_fused)
                _, score_fused_probs = encoder(img_fused)
                union_mask = torch.logical_or((score_tir > EPSILON), (score_oct > EPSILON)).float()
                # score_max = torch.maximum(score_tir, score_oct)
                # union_mask = torch.clamp(1.2 * score_max, 0, 1)

                # decoder backward
                restorm_loss_value = args.ssim_weight * (ssim_loss(tir_rec, img_tir) + ssim_loss(oct_rec, img_oct))
                pixel_loss_value = args.pixel_weight * (mse_loss(tir_rec, img_tir) + mse_loss(oct_rec, img_oct))
                # better_fusion_loss_value = args.fuse_weight * (mse_loss(score_fused_probs, union_mask) + ssim_loss(score_fused_probs, union_mask))
                better_fusion_loss_value = args.fuse_weight * mse_loss(score_fused_probs, union_mask)
                decoder_loss = restorm_loss_value + pixel_loss_value + better_fusion_loss_value
                decoder_loss.backward()
                Loss_restorm.append(restorm_loss_value.item())
                Loss_pixel.append(pixel_loss_value.item())
                Loss_fusion.append(better_fusion_loss_value.item())

                # optimize decoder
                optimizer.step()

                logging.info(
                    f"[Iter{i}] restorm: {restorm_loss_value.item():.5f} pixel: {pixel_loss_value.item():.5f} fusion: {better_fusion_loss_value.item():.5f}"
                )
                writer.add_scalar("train_loss/restorm", restorm_loss_value.item(), epoch * len(trainloader) + i)
                writer.add_scalar("train_loss/pixel", pixel_loss_value.item(), epoch * len(trainloader) + i)
                writer.add_scalar("train_loss/fusion", better_fusion_loss_value.item(), epoch * len(trainloader) + i)

        end_epoch = time.time()

        # logging.info cost
        logging.info(f"epoch: {epoch} time_taken: {end_epoch - start_epoch:.3f}")
        if args.training_encoder:
            quality_mean = np.mean(np.array(Loss_quality))
            ssim_mean = np.mean(np.array(Loss_ssim))
            logging.info(f"[Train loss] quality: {quality_mean:.5f} ssim: {ssim_mean:.5f}")
            train_loss = quality_mean + ssim_mean
        else:
            restorm_mean = np.mean(np.array(Loss_restorm))
            pixel_mean = np.mean(np.array(Loss_pixel))
            fusion_mean = np.mean(np.array(Loss_fusion))
            logging.info(f"[Train loss] restorm: {restorm_mean:.5f} pixel: {pixel_mean:.5f} fusion: {fusion_mean:.5f}")
            train_loss = restorm_mean + pixel_mean + fusion_mean
        writer.add_scalar("train_loss", train_loss, epoch)
        logging.info(f"[Train loss] {train_loss :.5f} minimum: {loss_minimum :.5f}")
        if train_loss < loss_minimum:
            loss_minimum = train_loss
            best_epoch = epoch

        # Get loss on the test set
        test_loss = test(encoder, decoder, testloader, epoch, write_result=True)
        writer.add_scalar("test_loss", test_loss, epoch)
        logging.info(f"[Test loss] {test_loss:.5f}")

        # Condition for reduce lr and early stopping
        if epoch - best_epoch > args.patience:
            args.lr /= 10
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

    if args.is_train:
        # Split dataset
        img_paths_tir = glob(join_path(tir_data_path, "*.bmp"))
        logging.info(f"Dataset size: {len(img_paths_tir)}")
        random.shuffle(img_paths_tir)
        test_paths = img_paths_tir[: args.test_num]
        train_paths = img_paths_tir[args.test_num :]
        train_set = FingerPrint(train_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=True, with_score=True)
        test_set = FingerPrint(test_paths, tir_data_path, oct_data_path, image_size=args.image_size, is_train=False, with_score=True)
        logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Train models
        encoder, decoder = train(trainloader, testloader)
        models_dir = join_path(args.output_dir, "models")
        create_dirs(models_dir)
        save_model(join_path(models_dir, f"final_enc.pth"), encoder)
        save_model(join_path(models_dir, f"final_dec.pth"), decoder)

        # train step
        # encoder, decoder = train_step(trainloader, testloader)
        # models_dir = join_path(args.output_dir, "models")
        # create_dirs(models_dir)
        # if args.training_encoder:
        #     save_model(join_path(models_dir, f"final_enc.pth"), encoder)
        # else:
        #     save_model(join_path(models_dir, f"final_dec.pth"), decoder)
    else:
        encoder = CDDFuseEncoder()
        decoder = CDDFuseDecoder()
        enc_model_path = args.pretrain_weight + "_enc.pth"
        dec_model_path = args.pretrain_weight + "_dec.pth"
        load_model(enc_model_path, encoder)
        load_model(dec_model_path, decoder)

        img_paths_tir = glob(join_path(tir_data_path, "*.bmp"))
        datasets = FingerPrint(img_paths_tir, tir_data_path, oct_data_path, is_train=False, with_score=True)
        testloader = DataLoader(datasets, batch_size=1, shuffle=False)
        test(encoder, decoder, testloader, 0, write_result=True)
