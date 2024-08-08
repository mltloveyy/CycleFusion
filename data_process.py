import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from utils import color_invert, read_image, save_tensor

img_tir_ext = ".bmp"
img_oct_ext = ".bmp"
score_tir_ext = "_Q.jpg"
score_oct_ext = "_Q.jpg"


class Augmentation:
    def __init__(
        self,
        image_size: int,
        mean: list = [0, 0, 0],
        std: list = [1.0, 1.0, 1.0],
    ):
        super().__init__()
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.normalize = v2.Normalize(mean, std)
        self.random_geometry = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                # v2.RandomRotation(degrees=180, interpolation=v2.InterpolationMode.BILINEAR),
                v2.RandomCrop(size=(image_size, image_size)),
            ]
        )

    def process(self, data, is_train=False):
        data = self.to_tensor(data)
        data = self.normalize(data[:2]) + data[2:]
        if is_train:
            data = self.random_geometry(data)
        return data


class FingerPrint(Dataset):
    def __init__(
        self,
        img_paths_tir: str,
        tir_root_dir: str,
        oct_root_dir: str,
        image_size: int = 256,
        is_train: bool = True,
        with_score: bool = False,
    ):
        super().__init__()
        self.all_paths = []
        self.filenames = []
        self.is_train = is_train
        self.with_score = with_score
        self.augmentation = Augmentation(image_size=image_size, mean=[0], std=[1])

        for img_path_tir in img_paths_tir:
            img_tir_basename = os.path.basename(img_path_tir)
            img_oct_basename = img_tir_basename.replace(img_tir_ext, img_oct_ext)
            score_tir_basename = img_tir_basename.replace(img_tir_ext, score_tir_ext)
            score_oct_basename = img_tir_basename.replace(img_tir_ext, score_oct_ext)
            img_path_oct = os.path.join(oct_root_dir, img_oct_basename)
            score_path_tir = os.path.join(tir_root_dir, score_tir_basename)
            score_path_oct = os.path.join(oct_root_dir, score_oct_basename)
            self.filenames.append(img_tir_basename.split(".")[0])

            if self.with_score:
                if os.path.exists(img_path_oct) and os.path.exists(score_path_tir) and os.path.exists(score_path_oct):
                    self.all_paths.append([img_path_tir, img_path_oct, score_path_tir, score_path_oct])
            else:
                if os.path.exists(img_path_oct):
                    self.all_paths.append([img_path_tir, img_path_oct])

    def __getitem__(self, index):
        img_tir = read_image(self.all_paths[index][0], True)
        img_oct = read_image(self.all_paths[index][1], True)
        # img_tir = color_invert(img_tir)
        # img_oct = color_invert(img_oct)
        if self.with_score:
            score_tir = read_image(self.all_paths[index][2])
            score_oct = read_image(self.all_paths[index][3])
            return self.augmentation.process((img_tir, img_oct, score_tir, score_oct), is_train=self.is_train)
        else:
            return self.augmentation.process((img_tir, img_oct), is_train=self.is_train)

    def __len__(self):
        return len(self.all_paths)


if __name__ == "__main__":
    from glob import glob

    tir_data_path = "images/dataset2/tir"
    oct_data_path = "images/dataset2/oct"
    img_paths_tir = glob(os.path.join(tir_data_path, "*.bmp"))
    datasets = FingerPrint(img_paths_tir, tir_data_path, oct_data_path, is_train=True, with_score=True)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False)

    test_num = 2
    for epoch in range(2):
        for i, data in enumerate(dataloader):
            save_tensor(data[0], f"{epoch}_{i}_tir_img.jpg")
            save_tensor(data[1], f"{epoch}_{i}_oct_img.jpg")
            save_tensor(data[2], f"{epoch}_{i}_tir_Q.jpg")
            save_tensor(data[3], f"{epoch}_{i}_oct_Q.jpg")
            if i > test_num:
                break
