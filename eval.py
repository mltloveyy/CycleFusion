import os
import time

import torch

from decoder import CDDFuseDecoder, DenseFuseDecoder
from encoder import CDDFuseEncoder, DenseFuseEncoder
from fusion import weight_fusion
from utils import *

device = torch.device("cuda:0")


def eval_quality(encoder, dir_input, dir_output):
    cost = 0.0
    count = 0
    with torch.no_grad():
        for img_name in os.listdir(dir_input):
            if not img_name.endswith(".bmp"):
                continue

            path_input = join_path(dir_input, img_name)
            path_output = join_path(dir_output, img_name)

            img = read_image(path_input).transpose(2, 0, 1)
            input = torch.unsqueeze(torch.from_numpy(img) / 255.0, dim=0).to(device)

            start = time.time()
            _, q_probs = encoder(input)
            cost += time.time() - start
            count += 1

            save_tensor(q_probs.cpu().detach(), path_output)
    return cost / count


def eval_restore(encoder, decoder, dir_input, dir_output):
    cost = 0.0
    count = 0
    with torch.no_grad():
        for img_name in os.listdir(dir_input):
            if not img_name.endswith(".bmp"):
                continue

            path_input = join_path(dir_input, img_name)
            path_output = join_path(dir_output, img_name)

            img = read_image(path_input).transpose(2, 0, 1)
            input = torch.unsqueeze(torch.from_numpy(img) / 255.0, dim=0).to(device)

            start = time.time()
            f, _ = encoder(input)
            output = decoder(f)
            cost += time.time() - start
            count += 1

            save_tensor(output.cpu().detach(), path_output)
    return cost / count


def eval_fuse(encoder, decoder, dir_tir, dir_oct, dir_output, fuse_type: str):
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
            f_tir, q_probs_tir = encoder(t_tir)
            f_oct, q_probs_oct = encoder(t_oct)
            f_fused = weight_fusion(f_tir, f_oct, q_probs_tir, q_probs_oct, strategy_type=fuse_type)
            t_fused = decoder(f_fused)
            _, q_probs_fused = encoder(t_fused)
            cost += time.time() - start
            count += 1

            path_output = join_path(dir_output, img_name)
            img_outputs = torch.cat((t_tir, t_oct, t_fused), dim=-1)
            score_outputs = torch.cat((q_probs_tir, q_probs_oct, q_probs_fused), dim=-1)
            outputs = torch.cat((img_outputs, score_outputs), dim=-2).cpu()
            save_tensor(outputs.detach(), path_output)
    return cost / count


if __name__ == "__main__":

    model_base = "output/20240805_173253/models/final"

    encoder = CDDFuseEncoder()
    load_model(model_base + "_enc.pth", encoder)
    encoder.to(device)
    encoder.eval()

    decoder = CDDFuseDecoder()
    load_model(model_base + "_dec.pth", decoder)
    decoder.to(device)
    decoder.eval()

    data_dir = "images/dataset4/"
    dir_tir = data_dir + "/tir"
    dir_oct = data_dir + "/oct"
    dir_output = data_dir + "/ours"
    create_dirs(dir_output)

    # fuse
    fuse_type = "pow"  # add, pow, exp
    cost = eval_fuse(encoder, decoder, dir_tir, dir_oct, dir_output, fuse_type)

    # restore
    # cost = eval_restore(encoder, decoder, dir_tir, dir_output)

    # quality
    # cost = eval_quality(encoder, dir_tir, dir_output)

    print(f"Average time: {cost}s")


# if __name__ == "__main__":
#     dir_input = "images/LowQualityFingerprint738/TIR"
#     dir_output = "images/LowQualityFingerprint738/TIR_enhanced"
#     create_dirs(dir_output)

#     for i, img_name in enumerate(os.listdir(dir_input)):
#         if not img_name.endswith(".bmp"):
#             continue

#         path_input = join_path(dir_input, img_name)
#         path_fused = join_path(dir_output, img_name)

#         img_input = cv2.imread(path_input, cv2.IMREAD_UNCHANGED) / 255.0
#         img_output = np.log(1 + img_input)
#         img_output = (img_output * 255).astype(np.uint8) + 60

#         cv2.imwrite(path_fused, img_output)
