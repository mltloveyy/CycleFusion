import os
import time

import torch

from networks import QualityFuser
from utils import *

device = torch.device("cuda:0")


def eval_quality(model: QualityFuser, dir_input, dir_output):
    cost = 0.0
    count = 0
    with torch.no_grad():
        for img_name in os.listdir(dir_input):
            if not img_name.endswith(".bmp"):
                continue

            path_input = join_path(dir_input, img_name)

            img = read_image(path_input).transpose(2, 0, 1)
            input = torch.from_numpy(img / 255.0).unsqueeze(0).to(device)

            start = time.time()
            q_probs = model.encoder(input)[-1]
            cost += time.time() - start
            count += 1

            path_output = join_path(dir_output, img_name)
            save_tensor(q_probs.cpu().detach(), path_output)
    return cost / count


def eval_restore(model: QualityFuser, dir_input, dir_output):
    cost = 0.0
    count = 0
    with torch.no_grad():
        for img_name in os.listdir(dir_input):
            if not img_name.endswith(".bmp"):
                continue

            path_input = join_path(dir_input, img_name)

            img = read_image(path_input).transpose(2, 0, 1)
            input = torch.from_numpy(img / 255.0).unsqueeze(0).to(device)

            start = time.time()
            if model.use_hybrid:
                f_base, f_detail, _ = model.encoder(input)
                output = model.decoder(f_base, f_detail)
            else:
                f, _ = model.encoder(input)
                output = model.decoder(f)
            cost += time.time() - start
            count += 1

            path_output = join_path(dir_output, img_name)
            save_tensor(output.cpu().detach(), path_output)
    return cost / count


def eval_fuse(model: QualityFuser, dir_tir, dir_oct, dir_output):
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
            t_tir = torch.from_numpy(img_tir / 255.0).unsqueeze(0).to(device)
            t_oct = torch.from_numpy(img_oct / 255.0).unsqueeze(0).to(device)

            start = time.time()
            model.encode(t_tir, t_oct)
            img_fused, _ = model.fuse()
            cost += time.time() - start
            count += 1

            path_output = join_path(dir_output, img_name)
            save_tensor(img_fused.cpu().detach(), path_output)
    return cost / count


if __name__ == "__main__":
    path_model = "output/20240814_024325/models/epoch40"
    if not os.path.exists(path_model):
        print(f"model path:{path_model} is not exists")
        exit()
    dir_data = "experiment/raw_data/738"
    dir_output = "experiment/results/738_cdd_add_new2"
    create_dirs(dir_output)

    model = QualityFuser(
        network_type="cddfuse",
        fuse_type="feature",
        with_quality=True,
        with_reval=False,
    )
    model.load(path_model)
    model.to("eval", device)

    dir_tir = dir_data + "/tir"
    dir_oct = dir_data + "/oct"

    # fuse
    cost = eval_fuse(model, dir_tir, dir_oct, dir_output)

    # restore
    # cost = eval_restore(model, dir_tir, dir_output)

    # quality
    # cost = eval_quality(encoder, dir_tir, dir_output)

    print(f"Average time: {cost}s")
