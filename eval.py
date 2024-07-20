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


def eval_quality(enc_path, image, output):
    encoder = CDDFuseEncoder()

    load_model(enc_path, encoder)
    encoder.eval()

    img = read_image(image).transpose(2, 0, 1)
    img_tensor = torch.from_numpy(img) / 255.0
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    _, score_probs = encoder(img_tensor, True)
    save_tensor(score_probs.detach(), output)


if __name__ == "__main__":
    enc_model = "output/20240719_094014/models/epoch16_enc.pth"
    img = "output/20240719_094014/fuse_0.jpg"
    output = "output/20240719_094014/fuse_0_score.jpg"
    eval_quality(enc_model, img, output)
