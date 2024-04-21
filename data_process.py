from glob import glob
from os.path import basename, exists, join, splitext

import torch
import torchvision.transforms as v2
from matplotlib.image import imread
from torch.utils.data import Dataset


class PreProcess:
    def __init__(self, image_size: int, mean: list, std: list):
        super().__init__()
        self.size = image_size
        self.mean = mean
        self.std = std

    def train(self):
        return v2.Compose(
            [
                v2.Resize(size=self.size),
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomAffine(degrees=179, translate=(0.2, 0.2)),
                v2.ToTensor(),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def test(self):
        return v2.Compose(
            [
                v2.Resize(size=self.size),
                v2.ToTensor(),
                v2.Normalize(mean=self.mean, std=self.std),
            ]
        )


class FingerPrint(Dataset):

    def __init__(self, tir_root_dir: str, oct_root_dir: str, with_score: False) -> None:
        super().__init__()
        img_paths_tir = glob(join(tir_root_dir, "*.jpg"))
        self.all_paths = []
        self.with_score = with_score
        for img_path_tir in img_paths_tir:
            base_name = splitext(basename(img_path_tir))[0]
            score_path_tir = join(tir_root_dir, base_name + ".bmp")
            img_path_oct = join(oct_root_dir, base_name + ".jpg")
            score_path_oct = join(oct_root_dir, base_name + ".bmp")
            if exists(score_path_tir) and exists(img_path_oct) and exists(score_path_oct):
                path_dict = [img_path_tir, score_path_tir, img_path_oct, score_path_oct]
                self.all_paths.append(path_dict)

    def __getitem__(self, index):
        img_tir = torch.from_numpy(imread(self.all_paths[index][0]))
        img_oct = torch.from_numpy(imread(self.all_paths[index][2]))
        if self.with_score:
            score_tir = torch.from_numpy(imread(self.all_paths[index][1]))
            score_oct = torch.from_numpy(imread(self.all_paths[index][3]))
            return img_tir, img_oct, score_tir, score_oct
        else:
            return img_tir, img_oct
