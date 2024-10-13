import os

import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import roc_curve

config = {
    "font.family": "Times New Roman",
    "font.size": 14.0,
    "axes.unicode_minus": False,
}  # 设置字体类型
rcParams.update(config)

# contrast
contrast_methods = [
    "TIR.txt",
    "OCT.txt",
    "DTNP-NSCT.txt",
    "Darlow's fusion.txt",
    "CSMCA-SHD.txt",
    "DenseFuse.txt",
    "CDDFuse.txt",
    "Proposed.txt",
]
contrast_show = [
    "External",
    "Internal",
    "DTNP-NSCT",
    "Darlow's fusion",
    "CSMCA-SHD",
    "DenseFuse",
    "CDDFuse",
    "Proposed",
]

# ablation
ablation_methods = [
    "cddfuse_add.txt",
    "cddfuse_weight.txt",
    "cddfuse_gradloss.txt",
    "densefuse_network_better.txt",
    "cddfuse(onlygfe)_network_better.txt",
    "cddfuse(onlydfe)_network_better.txt",
    "cddfuse_network_better.txt",
]
ablation_show = [
    "w/o quality-driven",
    "w/o fusion strategy",
    "w/o quality loss",
    "hybrid Enc → CNN Enc",
    "w/o DFE",
    "w/o GFE",
    "Proposed",
]

# mixture
mixture_methods = ["Internal&External.txt", "External&fusion.txt", "Internal&fusion.txt"]
mixture_show = [m.split(".")[0] for m in mixture_methods]

# discussion
discussion_methods = ["ocl.txt", "shd.txt"]
discussion_show = ["w/ OCL", "w/ SHD"]

# matplotlib
colors = ["blue", "green", "deeppink", "cyan", "magenta", "orange", "limegreen", "royalblue"]  # colors = plt.cm.viridis(np.linspace(0, 1, len(methods)))
marker_types = ["o", "v", "x", "s", "*", "^", "p", "d"]


if __name__ == "__main__":

    benefitPath = r"D:\code\pytorch\CycleFusion\experiments\det\contrast_1208"
    labelPath = r"D:\code\pytorch\CycleFusion\experiments\det\label_1208.txt"

    methods = contrast_methods
    show = contrast_show

    plt.figure(figsize=(8, 6))

    # label
    with open(labelPath, "r") as f:
        content = f.read()
        labels = content.split("\n")
        labels = [int(item) for item in labels if item.strip().isdigit()]

    for i, method in enumerate(methods):
        name = show[i]
        lists = []
        with open(os.path.join(benefitPath, method), "r") as f:
            content = f.read()
            contents = content.split("\n")
            contents = [int(item) for item in contents if item.strip().isdigit()]
        fpr, tpr, _ = roc_curve(labels, contents)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        frr = 1 - tpr
        plt.plot(fpr, frr, label=f"{name}", lw=1, color=colors[i], marker=marker_types[i], ms=3)
        print(f"{name}: {eer:.5f}")

    # EER基准线
    EER_x = [0, 1]
    EER_y = [0, 1]
    plt.plot(EER_x, EER_y, "r--")

    # plt.title("DET Curve")
    plt.legend(loc="lower left")  # upper right, lower left
    plt.xlim([0.00001, 0.1])
    plt.ylim([0.0001, 0.1])
    # plt.xlim([0.0001, 1])
    # plt.ylim([0.001, 1])

    plt.xlabel("FMR")
    plt.ylabel("FNMR")  # 可以使用中文，但需要导入一些库即字体

    # show with log
    plt.xscale("log")
    plt.yscale("log")
    plt.grid(which="major", color="grey")  # 显示主要网格线
    plt.grid(which="minor", color="lightgrey")  # 显示次要网格线

    plt.show()
    # plt.savefig("DET_curve.png")
