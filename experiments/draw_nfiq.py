import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import rcParams
from matplotlib.lines import Line2D

config = {
    "font.family": "Times New Roman",
    "font.size": 13.0,
    "axes.unicode_minus": False,
}  # 设置字体类型  # 解决负号无法显示的问题
rcParams.update(config)

if __name__ == "__main__":
    file_path = "nfiq/summary_contrast.xlsx"
    sheet_name = "1208"

    df = pd.read_excel(file_path, sheet_name=sheet_name)

    df = df.drop(["NSST"], axis=1)

    # 获取DataFrame的列名和数据
    labels = df.columns
    data = df.values  # 直接使用整个DataFrame的数据

    plt.figure(figsize=(7, 6))

    # 绘制箱型图
    plt.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        widths=0.7,
        autorange=True,
        showmeans=True,
        flierprops=dict(marker="D", markerfacecolor="black", markersize=3),
        meanprops=dict(marker="s", markeredgecolor="black", markerfacecolor="white", markersize=4),
        boxprops=dict(facecolor="cadetblue"),
        medianprops=dict(color="black"),
    )

    # 图例
    legend_elements = [
        Line2D([0], [0], marker="s", markersize=4, linestyle="none", markerfacecolor="none", markeredgecolor="black", label="Mean"),
        Line2D([0], [0], marker="D", markersize=3, linestyle="none", color="k", label="Outlier"),
        Line2D([0, 1], [0.5, 0.5], linestyle="-", color="k", label="Median line"),
    ]
    plt.legend(handles=legend_elements, loc="upper right", bbox_to_anchor=(1, 1))  # 将图例放置在左上角

    # 添加标题和标签（可选）
    # plt.title(f"Boxplot for {sheet_name}")
    plt.ylabel("NFIQ Score")

    # 如果x轴标签太长或重叠，可以旋转它们
    plt.xticks(rotation=45)

    # 显示图形
    plt.tight_layout()
    plt.grid(False)
    plt.ylim([0, 120])
    plt.show()
    # plt.savefig(f"{sheet_name}.png")
