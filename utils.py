from glob import glob
from os import makedirs
from os.path import exists, join

import torch
from matplotlib.image import imsave


def is_exists(path: str) -> bool:
    if path is not None:
        return exists(path)
    else:
        return False


def create_dirs(path: str):
    return makedirs(path)


def join_path(path: str, subpath: str) -> str:
    return join(path, subpath)


def save_tensor(tensor: torch.Tensor, path: str):
    array = tensor.clamp(0, 1).numpy()
    array = array.astype("uint8")
    imsave(path, array)
    print(f"save tensor at {path}")


def load_model(path, model):
    if is_exists(path):
        model.load_state_dict(torch.load(path))
        print(f"load model state dict from {path}")


def save_model(path, model):
    torch.save(model.state_dict(), path)
    print(f"save model state dict at {path}")
