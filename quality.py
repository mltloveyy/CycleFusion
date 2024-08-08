import cv2
import numpy as np
import torch
from scipy.ndimage import correlate

EPSILON = 1e-6


def get_mask(image: np.array, morph_kernel: int = 5, return_draw: bool = False) -> np.array:
    _, binary = cv2.threshold(image, 130, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # 闭操作(先膨胀后腐蚀)
    close = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)))
    close = cv2.erode(close, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel)))

    contours, _ = cv2.findContours(close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        if return_draw:
            mask = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            mask = np.zeros(image.shape, dtype=np.float32)
        for i, c in enumerate(contours):
            if cv2.contourArea(c) > 500:
                if return_draw:
                    cv2.drawContours(mask, contours, i, [0, 0, 255])
                else:
                    cv2.fillPoly(mask, [c], (1.0))
    else:
        mask = np.ones(image.shape).astype(np.float32)

    return mask


def calc_quality(image: np.array, mask: np.array = None, morph_kernel: int = 5) -> np.array:
    assert image.ndim == 2
    assert image.dtype == np.uint8
    grady, gradx = np.gradient(image)
    # x = x if x < 64, x = x - 128 if 64 <= x < 192, x = x - 256 if x >= 192
    grady = np.where(grady < 64, grady, grady - 128)
    grady = np.where(grady < 64, grady, grady - 128)
    gradx = np.where(gradx < 64, gradx, gradx - 128)
    gradx = np.where(gradx < 64, gradx, gradx - 128)

    # compute covariance matrix
    a = gradx * gradx
    b = grady * grady
    c = gradx * grady
    weight = np.ones((10, 10), np.float32)
    a = correlate(a, weight)
    b = correlate(b, weight)
    c = correlate(c, weight)

    # compute the eigenvalues
    eigv_max = ((a + b) + np.sqrt(pow(a - b, 2) + 4 * pow(c, 2))) / 2.0
    eigv_min = ((a + b) - np.sqrt(pow(a - b, 2) + 4 * pow(c, 2))) / 2.0

    # compute mask
    if mask is None:
        mask = get_mask(image, morph_kernel)
        mask = cv2.GaussianBlur(mask * 1.0, (7, 7), 0)

    # compute ocl
    ocl = np.where(eigv_max < EPSILON, EPSILON, 1.0 - eigv_min / (eigv_max + EPSILON))
    ocl = ocl * mask

    # compute sc
    sc = pow(eigv_max - eigv_min, 2) / pow(eigv_max + eigv_min + EPSILON, 2)
    sc = sc * mask

    # compute coh
    coh = (eigv_max - eigv_min) / (eigv_max + eigv_min + EPSILON)
    coh = coh * mask

    score = sc #ocl

    return score


def calc_quality_torch(tensor: torch.Tensor) -> torch.Tensor:
    array = tensor.cpu().numpy()
    score = [[calc_quality(np.squeeze(a))] for a in array]
    score = torch.Tensor(np.array(score)).cuda()
    return score


if __name__ == "__main__":
    import os

    input_dir = "images/dataset4/tir"
    kernel = 5  # tir
    # kernel = 5  # oct

    # mask
    # for file in os.listdir(input_dir):
    #     if file[-3:] == "bmp":
    #         file_path = os.path.join(input_dir, file)
    #         image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    #         # gen mask
    #         mask = get_mask(image, morph_kernel=kernel)
    #         mask = (mask * 255).astype(np.uint8)
    #         cv2.imwrite(file_path.replace(".bmp", "_M.jpg"), mask)
    #         # gen mask contour
    #         mask = get_mask(image, morph_kernel=kernel, return_draw=True)
    #         cv2.imwrite(file_path.replace(".bmp", "_C.jpg"), mask)

    # quality
    for file in os.listdir(input_dir):
        if file[-3:] == "bmp":
            file_path = os.path.join(input_dir, file)
            image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
            score = calc_quality(image, morph_kernel=kernel)
            score = (score * 255).astype(np.uint8)
            cv2.imwrite(file_path.replace(".bmp", "_Q.jpg"), score)

    # calc quality with torch
    # tensor = torch.rand(size=(4, 1, 256, 256), device="cuda")
    # score = calc_quality_torch(tensor)
