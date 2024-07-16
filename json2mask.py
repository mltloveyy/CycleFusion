import json

from PIL import Image, ImageDraw


def load_labelme_json(json_path):
    """
    加载Labelme的JSON标注文件
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    return data


def polygons_to_mask(img_shape, polygons):
    """
    将多边形转换为mask
    :param img_shape: 图像的形状 (height, width)
    :param polygons: 标注的多边形列表，每个多边形是一个点的列表 [(x1, y1), (x2, y2), ...]
    :return: 二值mask
    """
    mask = Image.new("L", img_shape, 0)  # 创建一个黑色的图像
    draw = ImageDraw.Draw(mask)
    for poly in polygons:
        poly = [(int(round(x)), int(round(y))) for x, y in poly]
        draw.polygon(poly, outline=None, fill=255)  # 用白色填充多边形
    return mask


def process(json_path, output_path):
    """
    主函数，处理Labelme JSON并生成mask图像
    :param json_path: Labelme JSON标注文件的路径
    :param image_path: 原始图像路径（可选，仅用于验证mask）
    :param output_path: 输出mask图像的路径
    """
    data = load_labelme_json(json_path)
    img_shape = (data["imageWidth"], data["imageHeight"])
    polygons = [shape["points"] for shape in data["shapes"] if shape["shape_type"] == "polygon"]

    mask = polygons_to_mask(img_shape, polygons)
    mask.save(output_path)


if __name__ == "__main__":
    import os

    path = "images/raw/tir"
    for file in os.listdir(path):
        json_path = os.path.join(path, file)
        if json_path[-3:] == "bmp":
            continue
        output_path = json_path.replace("json", "jpg")
        process(json_path, output_path)
