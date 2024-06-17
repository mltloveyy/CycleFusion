import os

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from utils import color_invert, read_image, save_tensor

# shh
img_tir_ext = "TIR-gray1-EF-DA.bmp"
img_oct_ext = "LMTU-gray1-IF-DA.bmp"
score_tir_ext = "TIR-gray1-EF-DA.jpg"
score_oct_ext = "LMTU-gray1-IF-DA.jpg"


class Augmentation:
    def __init__(
        self,
        image_size: int,
        mean: list = [0, 0, 0],
        std: list = [1.0, 1.0, 1.0],
    ):
        super().__init__()
        self.to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        self.resize = v2.Resize((image_size, image_size), antialias=True)
        self.normalize = v2.Normalize(mean, std)
        self.random_geometry = v2.Compose(
            [
                v2.RandomHorizontalFlip(),
                v2.RandomVerticalFlip(),
                v2.RandomAffine(degrees=179, translate=(0.1, 0.1)),
            ]
        )

    def process(self, data, is_train=False):
        data = self.to_tensor(data)
        data = self.resize(data)
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
        is_train: bool = True,
        with_score: bool = False,
    ):
        super().__init__()
        self.all_paths = []
        self.is_train = is_train
        self.with_score = with_score
        self.augmentation = Augmentation(image_size=256, mean=[0], std=[1])

        for img_path_tir in img_paths_tir:
            img_tir_basename = os.path.basename(img_path_tir)
            # img_oct_basename = img_tir_basename.replace(img_tir_ext, img_oct_ext)
            # score_tir_basename = img_tir_basename.replace(img_tir_ext, score_tir_ext)
            # score_oct_basename = img_tir_basename.replace(img_tir_ext, score_oct_ext)
            img_oct_basename = img_tir_basename
            score_tir_basename = img_tir_basename.replace("bmp", "jpg")
            score_oct_basename = score_tir_basename
            img_path_oct = os.path.join(oct_root_dir, img_oct_basename)
            score_path_tir = os.path.join(tir_root_dir, score_tir_basename)
            score_path_oct = os.path.join(oct_root_dir, score_oct_basename)

            if self.with_score:
                if os.path.exists(img_path_oct) and os.path.exists(score_path_tir) and os.path.exists(score_path_oct):
                    self.all_paths.append([img_path_tir, img_path_oct, score_path_tir, score_path_oct])
            else:
                if os.path.exists(img_path_oct):
                    self.all_paths.append([img_path_tir, img_path_oct])

    def __getitem__(self, index):
        img_tir = color_invert(read_image(self.all_paths[index][0]))
        img_oct = color_invert(read_image(self.all_paths[index][1]))
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

    tir_data_path = "images/tir"
    oct_data_path = "images/oct"
    img_paths_tir = glob(os.path.join(tir_data_path, "*.bmp"))
    datasets = FingerPrint(img_paths_tir, tir_data_path, oct_data_path, is_train=True, with_score=True)
    dataloader = DataLoader(datasets, batch_size=1, shuffle=False)

    test_num = 2
    for epoch in range(2):
        for i, data in enumerate(dataloader):
            union_mask = torch.logical_or((data[2] > 0), (data[3] > 0))
            save_tensor(union_mask, f"union_mask_{epoch}_{i}.jpg")
            save_tensor(data[0], f"tir_img_{epoch}_{i}.jpg")
            save_tensor(data[1], f"oct_img_{epoch}_{i}.jpg")
            save_tensor(data[2], f"tir_mask_{epoch}_{i}.jpg")
            save_tensor(data[3], f"oct_mask_{epoch}_{i}.jpg")
            if i > test_num:
                break

    # rename image
    # path = "images/oct"
    # for filename in os.listdir(path):
    #     new_filename = filename.replace("W", "F")
    #     os.rename(os.path.join(path, filename), os.path.join(path, new_filename))
