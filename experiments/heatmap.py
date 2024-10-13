import matplotlib.pyplot as plt
import numpy as np
import torch

from encoder import CDDFuseEncoder
from utils import *

device = torch.device("cuda:0")


def heatmap(feature: torch.Tensor, lower=-0.5, upper=0.3):
    array = feature[0, :, :, :].cpu().numpy()
    array = np.mean(array, axis=0)
    # lower = np.min(array)
    array = np.where(array < lower, lower, array)
    # upper = np.max(array)
    array = np.where(array > upper, upper, array)
    array = (array - lower) / (upper - lower)  # normalize
    return array


if __name__ == "__main__":
    enc1 = CDDFuseEncoder()
    load_model(r"experiments\heatmap\wo_quality_enc.pth", enc1)
    enc1.to(device)
    enc1.eval()

    enc2 = CDDFuseEncoder()
    load_model(r"experiments\heatmap\final_enc.pth", enc2)
    enc2.to(device)
    enc2.eval()

    img_list = [
        r"experiments\heatmap\images\WRX-L2-3.bmp",
        r"experiments\heatmap\images\LYX-R2-1.bmp",
        r"experiments\heatmap\images\CJJ-R1-2.bmp",
    ]

    fig, axs = plt.subplots(3, 3, figsize=(14, 10))

    for i, img_path in enumerate(img_list):
        img = read_image(img_path).transpose(2, 0, 1)
        tensor = torch.unsqueeze(torch.from_numpy(img) / 255.0, dim=0).to(device)
        with torch.no_grad():
            f1, _ = enc1(tensor)
            f2, _ = enc2(tensor)

        score = read_image(img_path.replace(".bmp", "_Q.jpg"))
        heatmap1 = heatmap(f1)
        heatmap2 = heatmap(f2)

        axs[i, 0].imshow(score, cmap="gray")
        cax1 = axs[i, 1].imshow(heatmap1, cmap="jet", interpolation="nearest", vmin=0.58, vmax=0.64)
        fig.colorbar(cax1, ax=axs[i, 1], orientation="vertical", fraction=0.046, pad=0.04)
        cax2 = axs[i, 2].imshow(heatmap2, cmap="jet", interpolation="nearest", vmin=0, vmax=1)
        fig.colorbar(cax2, ax=axs[i, 2], orientation="vertical", fraction=0.046, pad=0.04)

        for ax in axs[i, :]:
            ax.axis("off")

    plt.tight_layout()
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.show()
