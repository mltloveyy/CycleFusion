import os

import numpy as np
import torch

from decoder import CDDFuseDecoder, DenseFuseDecoder
from encoder import CDDFuseEncoder, DenseFuseEncoder
from fusion import weight_fusion
from utils import *


def eval(encoder, decoder, t_tir, t_oct):

    f_tir, q_probs_tir = encoder(t_tir)
    f_oct, q_probs_oct = encoder(t_oct)
    f_fused = weight_fusion(f_tir, f_oct, q_probs_tir, q_probs_oct, strategy_type="power")
    t_fused = decoder(f_fused)
    return t_fused


if __name__ == "__main__":
    enc_path = "output/20240728_130931/models/epoch180_enc.pth"
    dec_path = "output/20240728_130931/models/epoch180_dec.pth"

    encoder = None
    decoder = None
    if os.path.exists(enc_path):
        encoder = CDDFuseEncoder()
        load_model(enc_path, encoder)
        encoder.eval()
    if os.path.exists(dec_path):
        decoder = CDDFuseDecoder()
        load_model(dec_path, decoder)
        decoder.eval()

    dir_tir = "images/dataset3/tir"
    dir_oct = "images/dataset3/oct"
    dir_output = "images/dataset3/validation"
    create_dirs(dir_output)

    for img_name in os.listdir(dir_tir):
        if not img_name.endswith(".bmp"):
            continue
        path_tir = join_path(dir_tir, img_name)
        path_oct = join_path(dir_oct, img_name)
        path_fused = join_path(dir_output, img_name)

        img_tir = read_image(path_tir).transpose(2, 0, 1)
        t_tir = torch.unsqueeze(torch.from_numpy(img_tir) / 255.0, dim=0)

        img_oct = read_image(path_oct).transpose(2, 0, 1)
        t_oct = torch.unsqueeze(torch.from_numpy(img_oct) / 255.0, dim=0)

        t_fused = eval(encoder, decoder, t_tir, t_oct)
        save_tensor(t_fused.detach(), path_fused)
