from os import makedirs
from os.path import exists, join

import numpy as np
import torch
from matplotlib.image import imread, imsave


def is_exists(path: str) -> bool:
    if path is not None:
        return exists(path)
    else:
        return False


def create_dirs(path: str):
    return makedirs(path)


def join_path(path: str, subpath: str) -> str:
    return join(path, subpath)


def read_image(path: str):
    image = imread(path)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    return image


def color_invert(image: np.array):
    return 255 - image


def save_tensor(tensor: torch.Tensor, path: str):
    array = torch.squeeze(tensor).clamp(0, 1).numpy()  # bchw->chw
    array = (array * 255).astype("uint8")  # float->uint8
    array = array.transpose(1, 2, 0)  # chw->hwc
    imsave(path, array)
    print(f"save tensor at {path}")


def load_model(path, model):
    if is_exists(path):
        model.load_state_dict(torch.load(path))
        print(f"load model state dict from {path}")


def save_model(path, model):
    torch.save(model.state_dict(), path)
    print(f"save model state dict at {path}")
