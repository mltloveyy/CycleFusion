import logging
import random
import time
from glob import glob

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader

from args import parser
from data_process import FingerPrint
from decoder import CDDFuseDecoder, DenseFuseDecoder
from encoder import CDDFuseEncoder, DenseFuseEncoder
from fusion import weight_fusion
from losses import *
from utils import *

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPSILON = 1e-6

# Create output dir
args = parser.parse_args()
current_time = time.strftime("%Y%m%d_%H%M%S")
args.output_dir = join_path(args.output_dir, current_time)
create_dirs(args.output_dir)

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
    decoder.eval()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            assert len(data) == 4
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)
            score_tir = data[2].to(device)
            score_oct = data[3].to(device)

            # run the model on the test set to predict
            f_tir, score_tir_probs = encoder(img_tir)
            f_oct, score_oct_probs = encoder(img_oct)
            f_fused = weight_fusion(f_tir, f_oct, score_tir_probs, score_oct_probs, strategy_type="power")
            tir_rec = decoder(f_tir)
            oct_rec = decoder(f_oct)
            img_fused = decoder(f_fused)
            _, score_fused_probs = encoder(img_fused)
            union_mask = torch.logical_or((score_tir < EPSILON), (score_oct < EPSILON))

            # calc loss
            quality_loss_value = mse_loss(score_tir_probs, score_tir) + mse_loss(score_oct_probs, score_oct)
            ssim_loss_value = args.ssim_weight * (ssim_loss(tir_rec, img_tir) + ssim_loss(oct_rec, img_oct))
            pixel_loss_value = mse_loss(tir_rec, img_tir) + mse_loss(oct_rec, img_oct)
            better_fusion_loss_value = mse_loss(score_fused_probs, union_mask)
            total_loss_value = quality_loss_value.item() + ssim_loss_value.item() + pixel_loss_value.item() + better_fusion_loss_value.item()
            test_loss += total_loss_value
            logging.debug(f"[Test loss] quality: {quality_loss_value.item():.5f} ssim: {ssim_loss_value.item():.5f} pixel: {pixel_loss_value.item():.5f} fusion: {better_fusion_loss_value.item():.5f}")

            # save test set results
            if write_result & epoch % args.save_result_interval == 0:
                logging.info(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                imgs_cmp = torch.cat((img_tir, tir_rec, img_oct, oct_rec), dim=-1).cpu()
                score_cmp = torch.cat((score_tir, score_tir_probs, score_oct, score_oct_probs), dim=-1).cpu()
                fuse_cmp = torch.cat((img_tir, img_oct, img_fused, score_fused_probs), dim=-1).cpu()

                save_tensor(imgs_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_img_cmp.jpg"))
                save_tensor(score_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_score_cmp.jpg"))
                save_tensor(fuse_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_fuse_cmp.jpg"))

    return test_loss / len(testloader)


def train(trainloader, testloader):
    # init models
    encoder = CDDFuseEncoder()
    decoder = CDDFuseDecoder()
    load_model(args.pretrain_weight + "_enc.pth", encoder)
    load_model(args.pretrain_weight + "_dec.pth", decoder)

    optimizer_en = Adam(encoder.parameters(), lr=args.lr)
    optimizer_de = Adam(decoder.parameters(), lr=args.lr)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    loss_minimum = 10000.0
    Loss_quality, Loss_ssim, Loss_pixel, Loss_fusion = [], [], [], []

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        logging.info(f"start training No.{epoch} epoch...")
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
            _, score_tir_probs = encoder(img_tir, True)
            _, score_oct_probs = encoder(img_oct, True)

            # encoder backward
            quality_loss_value = mse_loss(score_tir_probs, score_tir) + mse_loss(score_oct_probs, score_oct)
            encoder_loss = quality_loss_value
            encoder_loss.backward()
            Loss_quality.append(quality_loss_value.item())

            # optimize encoder
            optimizer_en.step()

            if i % args.critic == 0:
                # froze encoder & decoder
                for param in encoder.parameters():
                    param.requires_grad = False

                # zero the decoder parameter gradients
                optimizer_de.zero_grad()

                # decoder forward
                f_tir, score_tir_probs = encoder(img_tir, True)
                f_oct, score_oct_probs = encoder(img_oct, True)
                f_fused = weight_fusion(f_tir, f_oct, score_tir_probs, score_oct_probs, strategy_type="power")
                tir_rec = decoder(f_tir)
                oct_rec = decoder(f_oct)
                img_fused = decoder(f_fused)
                _, score_fused_probs = encoder(img_fused, True)
                union_mask = torch.logical_or((score_tir > EPSILON), (score_oct > EPSILON)).float()

                # decoder backward
                ssim_loss_value = args.ssim_weight * (ssim_loss(tir_rec, img_tir) + ssim_loss(oct_rec, img_oct))
                pixel_loss_value = mse_loss(tir_rec, img_tir) + mse_loss(oct_rec, img_oct)
                better_fusion_loss_value = mse_loss(score_fused_probs, union_mask)
                decoder_loss = ssim_loss_value + pixel_loss_value + better_fusion_loss_value
                decoder_loss.backward()
                Loss_ssim.append(ssim_loss_value.item())
                Loss_pixel.append(pixel_loss_value.item())
                Loss_fusion.append(better_fusion_loss_value.item())
                logging.info(f"[Iter{i}] quality: {quality_loss_value.item():.5f} ssim: {ssim_loss_value.item():.5f} pixel: {pixel_loss_value.item():.5f} fusion: {better_fusion_loss_value.item():.5f}")

                # optimize decoder
                optimizer_de.step()

                # unfroze encoder & decoder
                for param in encoder.parameters():
                    param.requires_grad = True

        end_epoch = time.time()

        # logging.info loss
        logging.info(f"epoch: {epoch+1} time_taken: {end_epoch - start_epoch:.3f}")
        logging.info(f"[Train loss] quality: {np.mean(np.array(Loss_quality)):.5f} ssim: {np.mean(np.array(Loss_ssim)):.5f} pixel: {np.mean(np.array(Loss_pixel)):.5f} fusion: {np.mean(np.array(Loss_fusion)):.5f}")

        # Get loss on the test set
        test_loss = test(encoder, decoder, testloader, epoch, write_result=True)
        if test_loss < loss_minimum:
            loss_minimum = test_loss
            best_epoch = epoch
        logging.info(f"[Test loss] {test_loss :.5f} minimum: {loss_minimum :.5f}")

        # Condition for early stopping
        if epoch - best_epoch > args.patience:
            logging.info(f"Early stopping at epoch {epoch+1}")
            break

        # Save checkpoints
        if epoch > 5 & epoch % args.save_interval == 0:
            models_dir = join_path(args.output_dir, "models")
            create_dirs(models_dir)
            save_model(join_path(models_dir, f"epoch{epoch+1}_enc.pth"), encoder)
            save_model(join_path(models_dir, f"epoch{epoch+1}_dec.pth"), decoder)

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
        train_set = FingerPrint(train_paths, tir_data_path, oct_data_path, is_train=True, with_score=True)
        test_set = FingerPrint(test_paths, tir_data_path, oct_data_path, is_train=False, with_score=True)
        logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Train models
        encoder, decoder = train(trainloader, testloader)

        # Save models
        models_dir = join_path(args.output_dir, "models")
        create_dirs(models_dir)
        save_model(join_path(models_dir, f"final_enc.pth"), encoder)
        save_model(join_path(models_dir, f"final_dec.pth"), decoder)
    else:
        encoder = DenseFuseEncoder()
        decoder = DenseFuseDecoder()
        enc_model_path = args.pretrain_weight + "_enc.pth"
        dec_model_path = args.pretrain_weight + "_dec.pth"
        load_model(enc_model_path, encoder)
        load_model(dec_model_path, decoder)

        img_paths_tir = glob(join_path(tir_data_path, "*.bmp"))
        datasets = FingerPrint(img_paths_tir, tir_data_path, oct_data_path, is_train=False, with_score=True)
        testloader = DataLoader(datasets, batch_size=1, shuffle=False)
        test(encoder, decoder, testloader, 0, write_result=True)
