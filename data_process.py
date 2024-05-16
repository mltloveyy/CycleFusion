from glob import glob
from os.path import basename, exists, join, splitext

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from utils import color_invert, read_image, save_tensor

img_tir_ext = "TIR-gray1-EF-DA.bmp"
img_oct_ext = "LMTU-gray1-IF-DA.bmp"
score_tir_ext = "TIR-gray1-EF-DA.bmp"
score_oct_ext = "LMTU-gray1-IF-DA.bmp"


class PreProcess:
    def __init__(
        self,
        image_size: int,
        mean: list = [0, 0, 0],
        std: list = [1.0, 1.0, 1.0],
    ):
        super().__init__()
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.resize = v2.Resize((image_size, image_size))
        self.normalize = v2.Normalize(mean, std)
        self.random_geometry = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomAffine(degrees=179, translate=(0.1, 0.1)),
            ]
        )

    def transforms(self, data, is_train=False):
        data = self.to_tensor(data)
        data = self.resize(data)
        data = self.normalize(data[:2]) + data[2:]
        if is_train:
            data = self.random_geometry(data)
        return data


class FingerPrint(Dataset):
    def __init__(self, tir_root_dir: str, oct_root_dir: str, with_score: bool = False) -> None:
        super().__init__()
        img_paths_tir = glob(join(tir_root_dir, "*.bmp"))
        self.all_paths = []
        self.with_score = with_score
        self.preprocess = PreProcess(image_size=256, mean=[0], std=[1])

        for img_path_tir in img_paths_tir:
            img_tir_basename = basename(img_path_tir)
            img_oct_basename = img_tir_basename.replace(img_tir_ext, img_oct_ext)
            score_tir_basename = img_tir_basename.replace(img_tir_ext, score_tir_ext)
            score_oct_basename = img_tir_basename.replace(img_tir_ext, score_oct_ext)
            img_path_oct = join(oct_root_dir, img_oct_basename)
            score_path_tir = join(tir_root_dir, score_tir_basename)
            score_path_oct = join(oct_root_dir, score_oct_basename)

            if self.with_score:
                if exists(img_path_oct) and exists(score_path_tir) and exists(score_path_oct):
                    path_dict = [img_path_tir, img_path_oct, score_path_tir, score_path_oct]
            else:
                if exists(img_path_oct):
                    path_dict = [img_path_tir, img_path_oct]
            self.all_paths.append(path_dict)

    def __getitem__(self, index):
        img_tir = color_invert(read_image(self.all_paths[index][0]))
        img_oct = color_invert(read_image(self.all_paths[index][1]))
        if self.with_score:
            score_tir = read_image(self.all_paths[index][2])
            score_oct = read_image(self.all_paths[index][3])
            return self.preprocess.transforms((img_tir, img_oct, score_tir, score_oct), True)
        else:
            return self.preprocess.transforms((img_tir, img_oct), True)

    def __len__(self):
        return len(self.all_paths)


if __name__ == "__main__":
    tir_data_path = "images/tir"
    oct_data_path = "images/oct"
    datasets = FingerPrint(tir_data_path, oct_data_path)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False)

    test_num = 5
    for i, data in enumerate(dataloader):
        save_tensor(data[0], f"img_tir_{i}.jpg")
        save_tensor(data[1], f"img_oct_{i}.jpg")
        if i > test_num:
            exit(0)
