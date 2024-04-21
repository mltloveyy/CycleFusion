import logging
import time

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from args import parser
from data_process import FingerPrint, PreProcess
from decoder import CDDFuseDecoder
from deformer import CnnRegisterer
from encoder import CDDFuseEncoder
from fusion import FeatureFusion
from losses import *
from quality import calc_quality
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
    filename=join_path(args.output_dir, current_time + ".log"),
    format="%(asctime)s - %(funcName)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

# Set transforms
transforms = PreProcess(args.image_size, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
preprocess_train = transforms.train()
preprocess_test = transforms.test()


def test(encoder, decoder, deformer, fuser, testloader, epoch, write_result=False):
    logging.info("start testing...")
    encoder.eval()
    decoder.eval()
    deformer.eval()
    fuser.eval()
    test_loss = 0.0

    with torch.no_grad():
        for i, data in enumerate(testloader):
            data = preprocess_test(data)
            assert len(data) == 2
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # run the model on the test set to predict
            _, d_f_tir = encoder(img_tir)
            _, d_f_oct = encoder(img_oct)
            rec_tir = decoder(detail_feature=d_f_tir)
            rec_oct = decoder(detail_feature=d_f_oct)
            img_tir_def, flow = deformer(d_f_tir, d_f_oct, img_tir)
            _, d_f_img_tir_def = encoder(img_tir_def)
            f_fused = fuser(d_f_img_tir_def, d_f_oct)
            img_fused = decoder(f_fused)
            q_tir = calc_quality(img_tir)
            q_oct = calc_quality(img_oct)
            q_fused = calc_quality(img_fused)

            # calc loss
            ssim_loss_value = args.ssim_weight * (ssim_loss(rec_tir, img_tir) + ssim_loss(rec_oct, img_oct))
            rec_loss_value = mse_loss(rec_tir, img_tir) + mse_loss(rec_oct, img_oct)
            ncc_loss_value = ncc_loss(img_tir_def, img_oct)
            diff_loss_value = mse_loss(img_tir_def, img_oct)
            grad_loss_value = grad_loss(flow)
            fuse_loss_value = args.fuse_weight * mse_loss(img_fused, img_tir) + (1 - args.fuse_weight) * mse_loss(
                img_fused, img_oct
            )
            q_max = torch.maximum(q_tir, q_oct)
            quality_loss_value = torch.mean((q_max - q_fused)[q_max > q_fused])
            regular_loss_value = args.regular_factor * (
                torch.mean(q_fused[q_fused > args.quality_thresh]) - args.quality_thresh
            )

            total_loss_value = (
                ssim_loss_value.item()
                + rec_loss_value.item()
                + ncc_loss_value.item()
                + diff_loss_value.item()
                + grad_loss_value.item()
                + fuse_loss_value.item()
                + quality_loss_value.item()
                - regular_loss_value.item()
            )
            test_loss += total_loss_value
            logging.debug(
                f"[Test loss] ssim: {ssim_loss_value.item():.5f} rec: {rec_loss_value.item():.5f} \
                    ncc: {ncc_loss_value.item():.5f} diff: {diff_loss_value.item():.5f} \
                    grad: {grad_loss_value.item():.5f} fuse: {fuse_loss_value.item():.5f} \
                    quality: {quality_loss_value.item():.5f} regular: {regular_loss_value.item():.5f}"
            )

            # save test set results
            if write_result & epoch % args.save_interval == 0:
                logging.info(f"save results at epoch={epoch}")
                save_dir = join_path(args.output_dir, "validations")
                create_dirs(save_dir)
                imgs_cmp = torch.cat(
                    (torch.cat((img_tir, rec_tir), dim=1), torch.cat((img_oct, rec_oct), dim=1)),
                    dim=1,
                ).cpu()
                deform_cmp = torch.cat((img_tir, img_oct, img_tir_def), dim=1).cpu()
                fuse_cmp = torch.cat((img_tir, img_oct, img_fused), dim=1).cpu()
                quality_cmp = torch.cat((q_tir, q_oct, q_fused), dim=1).cpu()

                save_tensor(imgs_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_img_cmp.jpg"))
                save_tensor(deform_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_deform_cmp.jpg"))
                save_tensor(fuse_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_fuse_cmp.jpg"))
                save_tensor(quality_cmp, join_path(save_dir, f"epoch_{epoch}_No.{i}_quality_cmp.jpg"))

    return test_loss / len(testloader)


def train(trainloader, testloader):
    # init models
    encoder = CDDFuseEncoder()
    decoder = CDDFuseDecoder()
    deformer = CnnRegisterer()
    fuser = FeatureFusion()

    enc_model_path = args.pretrain_weight + "_enc.pth"
    dec_model_path = args.pretrain_weight + "_dec.pth"
    def_model_path = args.pretrain_weight + "_def.pth"
    fuse_model_path = args.pretrain_weight + "_fuse.pth"

    load_model(enc_model_path, encoder)
    load_model(dec_model_path, decoder)
    load_model(def_model_path, deformer)
    load_model(fuse_model_path, fuser)

    optimizer1 = Adam(
        [
            {"params": encoder.parameters()},
            {"params": decoder.parameters()},
        ],
        lr=args.lr,
    )
    optimizer2 = Adam(deformer.parameters(), lr=args.lr)
    optimizer3 = Adam(fuser.parameters(), lr=args.lr)

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    deformer = deformer.to(device)
    fuser = fuser.to(device)

    loss_minimum = 10000.0
    loss_ssim = []
    loss_rec = []
    loss_ncc = []
    loss_diff = []
    loss_grad = []
    loss_fuse = []
    loss_quality = []
    loss_regular = []

    for epoch in range(args.epochs):  # loop over the dataset multiple times
        logging.info(f"start training No.{epoch+1} epoch...")
        start_epoch = time.time()

        for i, data in enumerate(trainloader):
            # get the inputs
            data = preprocess_train(data)
            assert len(data) == 2
            img_tir = data[0].to(device)
            img_oct = data[1].to(device)

            # zero the reconstructor parameter gradients
            optimizer1.zero_grad()

            # reconstructor forward
            logging.debug("training reconstructor...")
            _, d_f_tir = encoder(img_tir).detach()
            _, d_f_oct = encoder(img_oct).detach()
            rec_tir = decoder(detail_feature=d_f_tir).detach()
            rec_oct = decoder(detail_feature=d_f_oct).detach()

            # reconstructor backward
            ssim_loss_value = args.ssim_weight * (ssim_loss(rec_tir, img_tir) + ssim_loss(rec_oct, img_oct))
            rec_loss_value = mse_loss(rec_tir, img_tir) + mse_loss(rec_oct, img_oct)
            loss1 = ssim_loss_value + rec_loss_value
            loss1.backward()
            loss_ssim.append(ssim_loss_value.item())
            loss_rec.append(rec_loss_value.item())
            logging.debug(f"ssim_loss: {ssim_loss_value.item():.5f}")
            logging.debug(f"rec_loss: {rec_loss_value.item():.5f}")

            # optimize reconstructor
            optimizer1.step()

            if i % args.critic == 0:
                # zero the deformer parameter gradients
                optimizer2.zero_grad()

                # deformer forward
                logging.debug("training deformer...")
                _, d_f_tir = encoder(img_tir)
                _, d_f_oct = encoder(img_oct)
                img_tir_def, flow = deformer(d_f_tir, d_f_oct, img_tir)

                # deformer backward
                ncc_loss_value = ncc_loss(img_tir_def, img_oct)
                diff_loss_value = mse_loss(img_tir_def, img_oct)
                grad_loss_value = grad_loss(flow)
                loss2 = ncc_loss_value + diff_loss_value + grad_loss_value
                loss2.backward()
                loss_ncc.append(ncc_loss_value.item())
                loss_diff.append(diff_loss_value.item())
                loss_grad.append(grad_loss_value.item())
                logging.debug(f"ncc_loss: {ncc_loss_value.item():.5f}")
                logging.debug(f"diff_loss: {diff_loss_value.item():.5f}")
                logging.debug(f"grad_loss: {grad_loss_value.item():.5f}")

                # optimize deformer
                optimizer2.step()

                # zero the fuser parameter gradients
                optimizer3.zero_grad()

                _, d_f_img_tir_def = encoder(img_tir_def)
                f_fused = fuser(d_f_img_tir_def, d_f_oct)
                img_fused = decoder(f_fused)
                q_tir = calc_quality(img_tir)
                q_oct = calc_quality(img_oct)
                q_fused = calc_quality(img_fused)

                # fuser backward
                fuse_loss_value = args.fuse_weight * mse_loss(img_fused, img_tir) + (1 - args.fuse_weight) * mse_loss(
                    img_fused, img_oct
                )
                q_max = torch.maximum(q_tir, q_oct)
                quality_loss_value = torch.mean((q_max - q_fused)[q_max > q_fused])
                regular_loss_value = args.regular_factor * (
                    torch.mean(q_fused[q_fused > args.quality_thresh]) - args.quality_thresh
                )
                loss3 = fuse_loss_value + quality_loss_value - regular_loss_value
                loss3.backward()
                loss_fuse.append(fuse_loss_value.item())
                loss_quality.append(quality_loss_value.item())
                loss_regular.append(regular_loss_value.item())
                logging.debug(f"fuse_loss: {fuse_loss_value.item():.5f}")
                logging.debug(f"quality_loss: {quality_loss_value.item():.5f}")
                logging.debug(f"regular_loss: {regular_loss_value.item():.5f}")

                # optimize fuser
                optimizer3.step()

        end_epoch = time.time()

        # logging.info loss
        logging.info(f"epoch: {epoch+1} time_taken: {end_epoch - start_epoch:.3f}")
        logging.info(
            f"[Train loss] ssim: {np.mean(np.array(loss_ssim)):.5f} rec: {np.mean(np.array(loss_rec)):.5f} \
                ncc: {np.mean(np.array(loss_ncc)):.5f} diff: {np.mean(np.array(loss_diff)):.5f} \
                grad: {np.mean(np.array(loss_grad)):.5f}  fuse: {np.mean(np.array(loss_fuse)):.5f} \
                quality: {np.mean(np.array(loss_quality)):.5f} regular: {np.mean(np.array(loss_regular)):.5f}"
        )

        # Get loss on the test set
        test_loss = test(encoder, decoder, deformer, fuser, testloader, epoch + 1)
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
            save_model(join_path(models_dir, f"epoch{epoch+1}_def.pth"), deformer)
            save_model(join_path(models_dir, f"epoch{epoch+1}_fuse.pth"), fuser)

        logging.info("========================================")

    return encoder, decoder, deformer, fuser


if __name__ == "__main__":
    # Fetch dataset
    tir_data_path = join_path(args.data_dir, "tir")
    oct_data_path = join_path(args.data_dir, "oct")
    datasets = FingerPrint(tir_data_path, oct_data_path)
    logging.info(f"Dataset size: {len(datasets)}")

    if args.is_train:
        # Split dataset
        train_set, test_set = random_split(datasets, [args.train_ratio, 1 - args.train_ratio])
        logging.info(f"Train set size: {len(train_set)}, Test set size: {len(test_set)}")
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

        enc_model_path = args.pretrain_weight + "_enc.pth"
        dec_model_path = args.pretrain_weight + "_dec.pth"
        def_model_path = args.pretrain_weight + "_def.pth"
        fuse_model_path = args.pretrain_weight + "_fuse.pth"

        load_model(enc_model_path, encoder)
        load_model(dec_model_path, decoder)
        load_model(def_model_path, deformer)
        load_model(fuse_model_path, fuser)

        testloader = DataLoader(datasets, batch_size=1, shuffle=False)
        test(encoder, decoder, deformer, fuser, testloader, 0, write_result=True)
