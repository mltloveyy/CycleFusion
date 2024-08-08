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
from decoder import CDDFuseDecoder
from deformer import CnnRegisterer
from encoder import CDDFuseEncoder
from fusion import FeatureFusion
from losses import *
from quality import calc_quality_torch
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


def test(encoder, decoder, deformer, fuser, testloader, epoch, write_result=False):
    logging.info("start testing...")
    encoder.eval()
    decoder.eval()
    deformer.eval()
    fuser.eval()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # run the model on the test set to predict
            f_tir = encoder(img_tir)
            f_oct = encoder(img_oct)
            rec_tir = decoder(features=f_tir)
            rec_oct = decoder(features=f_oct)
            img_oct_def, flow = deformer(f_tir, f_oct, img_oct)
            f_img_oct_def = encoder(img_oct_def)
            f_fused = fuser(f_img_oct_def, f_tir)
            img_fused = decoder(f_fused)
            q_tir = calc_quality_torch(img_tir)
            q_oct = calc_quality_torch(img_oct)
            q_fused = calc_quality_torch(img_fused.detach())

            # calc loss
            ssim_loss_value = args.ssim_weight * (ssim_loss(rec_tir, img_tir) + ssim_loss(rec_oct, img_oct))
            rec_loss_value = mse_loss(rec_tir, img_tir) + mse_loss(rec_oct, img_oct)
            ncc_loss_value = ncc_loss(img_oct_def, img_tir)
            grad_loss_value = grad_loss(flow)
            fuse_loss_value = args.fuse_weight * mse_loss(img_fused, img_tir) + (1 - args.fuse_weight) * mse_loss(img_fused, img_oct)
            q_max = torch.maximum(q_tir, q_oct)
            quality_loss_value = torch.mean((q_max - q_fused)[q_max > q_fused])
            regular_loss_value = args.regular_factor * (torch.mean(q_fused[q_fused > args.quality_thresh]) - args.quality_thresh)

            total_loss_value = (
                ssim_loss_value.item()
                + rec_loss_value.item()
                + ncc_loss_value.item()
                + grad_loss_value.item()
                + fuse_loss_value.item()
                + quality_loss_value.item()
                - regular_loss_value.item()
            )
            test_loss += total_loss_value
            logging.info(
                f"[Test loss] ssim: {ssim_loss_value.item():.5f} rec: {rec_loss_value.item():.5f} ncc: {ncc_loss_value.item():.5f} grad: {grad_loss_value.item():.5f} fuse: {fuse_loss_value.item():.5f} quality: {quality_loss_value.item():.5f} regular: {regular_loss_value.item():.5f}"
            )

            # save test set results
            if write_result & epoch % args.save_interval == 0:
                logging.info(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                imgs_cmp = torch.cat((img_tir, rec_tir, img_oct, rec_oct), dim=-1).cpu()
                deform_cmp = torch.cat((img_tir, img_oct, img_oct_def), dim=-1).cpu()
                fuse_cmp = torch.cat((img_tir, img_oct, img_fused), dim=-1).cpu()
                quality_cmp = torch.cat((q_tir, q_oct, q_fused), dim=-1).cpu()

                save_tensor(imgs_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_img_cmp.jpg"))
                save_tensor(deform_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_deform_cmp.jpg"))
                save_tensor(fuse_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_fuse_cmp.jpg"))
                save_tensor(quality_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_quality_cmp.jpg"))

        return test_loss / len(testloader)


def train(trainloader, testloader):
    # init models
    encoder = CDDFuseEncoder()
    decoder = CDDFuseDecoder()
    deformer = CnnRegisterer(img_size=args.image_size)
    fuser = FeatureFusion()
    load_model(args.pretrain_weight + "_enc.pth", encoder)
    load_model(args.pretrain_weight + "_dec.pth", decoder)
    load_model(args.pretrain_weight + "_def.pth", deformer)
    load_model(args.pretrain_weight + "_fuse.pth", fuser)

    optimizer1 = Adam([{"params": encoder.parameters()}, {"params": decoder.parameters()}], lr=args.lr)
    optimizer2 = Adam([{"params": deformer.parameters()}, {"params": fuser.parameters()}], lr=args.lr)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    deformer = deformer.to(device)
    fuser = fuser.to(device)

    loss_minimum = 10000.0
    loss_ssim = []
    loss_rec = []
    loss_ncc = []
    loss_grad = []
    loss_fuse = []
    loss_quality = []
    loss_regular = []

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        logging.info(f"start training No.{epoch} epoch...")
        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            # get the inputs
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # zero the reconstructor parameter gradients
            optimizer1.zero_grad()

            # reconstructor forward
            f_tir = encoder(img_tir)
            f_oct = encoder(img_oct)
            rec_tir = decoder(features=f_tir)
            rec_oct = decoder(features=f_oct)

            # reconstructor backward
            ssim_loss_value = args.ssim_weight * (ssim_loss(rec_tir, img_tir) + ssim_loss(rec_oct, img_oct))
            rec_loss_value = mse_loss(rec_tir, img_tir) + mse_loss(rec_oct, img_oct)
            loss1 = ssim_loss_value + rec_loss_value
            loss1.backward()
            loss_ssim.append(ssim_loss_value.item())
            loss_rec.append(rec_loss_value.item())

            # optimize reconstructor
            optimizer1.step()

            if i % args.critic == 0:
                # froze encoder & decoder
                for param in encoder.parameters():
                    param.requires_grad = False
                for param in decoder.parameters():
                    param.requires_grad = False

                # zero the parameter gradients of deformer and fuser
                optimizer2.zero_grad()

                # deformer and fuser forward
                f_tir = encoder(img_tir)
                f_oct = encoder(img_oct)
                img_oct_def, flow = deformer(f_tir, f_oct, img_oct)
                f_img_oct_def = encoder(img_oct_def)
                f_fused = fuser(f_img_oct_def, f_tir)
                img_fused = decoder(f_fused)
                q_tir = calc_quality_torch(img_tir)
                q_oct = calc_quality_torch(img_oct)
                q_fused = calc_quality_torch(img_fused.detach())

                # deformer and fuser backward
                ncc_loss_value = ncc_loss(img_oct_def, img_tir)
                grad_loss_value = grad_loss(flow)
                fuse_loss_value = args.fuse_weight * mse_loss(img_fused, img_tir) + (1 - args.fuse_weight) * mse_loss(img_fused, img_oct)
                q_max = torch.maximum(q_tir, q_oct)
                quality_loss_value = torch.mean((q_max - q_fused)[q_max > q_fused])
                regular_loss_value = args.regular_factor * (torch.mean(q_fused[q_fused > args.quality_thresh]) - args.quality_thresh)
                loss2 = ncc_loss_value + grad_loss_value + fuse_loss_value + quality_loss_value - regular_loss_value
                loss2.backward()
                loss_ncc.append(ncc_loss_value.item())
                loss_grad.append(grad_loss_value.item())
                loss_fuse.append(fuse_loss_value.item())
                loss_quality.append(quality_loss_value.item())
                loss_regular.append(regular_loss_value.item())
                logging.info(
                    f"[Iter{i}] ssim: {ssim_loss_value.item():.5f} rec: {rec_loss_value.item():.5f} ncc: {ncc_loss_value.item():.5f} grad: {grad_loss_value.item():.5f} fuse: {fuse_loss_value.item():.5f} quality: {quality_loss_value.item():.5f} regular: {regular_loss_value.item():.5f}"
                )

                # optimize deformer and fuser
                optimizer2.step()

                # unfroze encoder & decoder
                for param in encoder.parameters():
                    param.requires_grad = True
                for param in decoder.parameters():
                    param.requires_grad = True

        end_epoch = time.time()

        # epoch loss
        mean_ssim = np.mean(np.array(loss_ssim))
        mean_rec = np.mean(np.array(loss_rec))
        mean_ncc = np.mean(np.array(loss_ncc))
        mean_grad = np.mean(np.array(loss_grad))
        mean_fuse = np.mean(np.array(loss_fuse))
        mean_quality = np.mean(np.array(loss_quality))
        mean_regular = np.mean(np.array(loss_regular))
        logging.info(f"epoch: {epoch} time_taken: {end_epoch - start_epoch:.3f}")
        logging.info(
            f"[Train loss] ssim: {mean_ssim:.5f} rec: {mean_rec:.5f} ncc: {mean_ncc:.5f} grad: {mean_grad:.5f} fuse: {mean_fuse:.5f} quality: {mean_quality:.5f} regular: {mean_regular:.5f}"
        )

        # Get loss on the test set
        test_loss = test(encoder, decoder, deformer, fuser, testloader, epoch, write_result=True)
        if test_loss < loss_minimum:
            loss_minimum = test_loss
            best_epoch = epoch
        logging.info(f"[Test loss] {test_loss :.5f} minimum: {loss_minimum :.5f}")

        # Condition for early stopping
        if epoch - best_epoch > args.patience:
            logging.info(f"Early stopping at epoch {epoch}")
            break

        # Save checkpoints
        if epoch > 5 & epoch % args.save_interval == 0:
            models_dir = join_path(args.output_dir, "models")
            create_dirs(models_dir)
            save_model(join_path(models_dir, f"epoch{epoch}_enc.pth"), encoder)
            save_model(join_path(models_dir, f"epoch{epoch}_dec.pth"), decoder)
            save_model(join_path(models_dir, f"epoch{epoch}_def.pth"), deformer)
            save_model(join_path(models_dir, f"epoch{epoch}_fuse.pth"), fuser)

        logging.info("========================================")

    return encoder, decoder, deformer, fuser


if __name__ == "__main__":
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
        train_set = FingerPrint(train_paths, tir_data_path, oct_data_path, is_train=True)
        test_set = FingerPrint(test_paths, tir_data_path, oct_data_path, is_train=False)
        logging.info(f"Train set size: {len(train_paths)}, Test set size: {len(test_paths)}")
        trainloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False)

        # Train models
        encoder, decoder, deformer, fuser = train(trainloader, testloader)

        # Save models
        models_dir = join_path(args.output_dir, "models")
        create_dirs(models_dir)
        save_model(join_path(models_dir, f"final_enc.pth"), encoder)
        save_model(join_path(models_dir, f"final_dec.pth"), decoder)
        save_model(join_path(models_dir, f"final_def.pth"), deformer)
        save_model(join_path(models_dir, f"final_fuse.pth"), fuser)
    else:
        encoder = CDDFuseEncoder()
        decoder = CDDFuseDecoder()
        deformer = CnnRegisterer()
        fuser = FeatureFusion()
        load_model(args.pretrain_weight + "_enc.pth", encoder)
        load_model(args.pretrain_weight + "_dec.pth", decoder)
        load_model(args.pretrain_weight + "_def.pth", deformer)
        load_model(args.pretrain_weight + "_fuse.pth", fuser)

        img_paths_tir = glob(join_path(tir_data_path, "*.bmp"))
        datasets = FingerPrint(img_paths_tir, tir_data_path, oct_data_path, is_train=False)
        testloader = DataLoader(datasets, batch_size=1, shuffle=False)
        test(encoder, decoder, deformer, fuser, testloader, 0, write_result=True)
