import os

import cv2
import numpy as np
import torch


def create_dirs(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def join_path(path: str, subpath: str) -> str:
    return os.path.join(path, subpath)


def read_image(path: str, boost: bool = False) -> np.array:
    image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if boost:
        image = cv2.convertScaleAbs(image, alpha=1.08, beta=0)
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    return image


def color_invert(image: np.array):
    return 255 - image


def save_tensor(tensor: torch.Tensor, path: str):
    array = torch.squeeze(tensor, dim=0).clamp(0, 1).numpy()  # bchw->chw
    array = (array * 255).astype(np.uint8)  # float->uint8
    array = array.transpose(1, 2, 0)  # chw->hwc
    cv2.imwrite(path, array)
    print(f"save tensor at {path}")


def load_model(path: str, model):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"load model state dict from {path}")


def save_model(path: str, model):
    torch.save(model.state_dict(), path)
    print(f"save model state dict at {path}")
