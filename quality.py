import cv2
import numpy as np
import torch
from scipy.ndimage import correlate

EPSILON = 1e-6


def get_mask(image: np.array) -> np.array:
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 70, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    open = cv2.erode(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))  # 腐蚀
    open = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))  # 膨胀
    contours, _ = cv2.findContours(open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) > 0:
        mask = np.zeros(image.shape, dtype=np.float32)
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.fillPoly(mask, [largest_contour], (1.0))
    else:
        mask = np.ones(image.shape).astype(np.float32)

    return mask


def calc_quality(image: np.array, mask: np.array) -> np.array:
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
    # mask = get_mask(image)
    # roi = cv2.GaussianBlur(mask, (5, 5), 0)

    # compute ocl
    ocl = np.where(eigv_max < EPSILON, EPSILON, 1.0 - eigv_min / (eigv_max + EPSILON))
    ocl = ocl * mask

    # compute sc
    sc = pow(eigv_max - eigv_min, 2) / pow(eigv_max + eigv_min + EPSILON, 2)
    sc = sc * mask

    # compute coh
    coh = (eigv_max - eigv_min) / (eigv_max + eigv_min + EPSILON)
    coh = coh * mask

    score = ocl

    return score


def calc_quality_torch(tensor: torch.Tensor) -> torch.Tensor:
    array = tensor.cpu().numpy()
    score = [[calc_quality(np.squeeze(a))] for a in array]
    score = torch.Tensor(np.array(score)).cuda()
    return score


if __name__ == "__main__":
    import os

    path = "images/raw/tir"

    # gen mask
    # for file in os.listdir(path):
    #     file_path = os.path.join(path, file)
    #     image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    #     mask = get_mask(image)
    #     mask = (mask * 255).astype(np.uint8)
    #     cv2.imwrite(file_path.replace(".bmp", ".jpg"), mask)

    # calc quality
    for file in os.listdir(path):
        bmp_path = os.path.join(path, file)
        if bmp_path[-3:] == "bmp":
            image = cv2.imread(bmp_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(bmp_path.replace("bmp", "jpg"), cv2.IMREAD_UNCHANGED)
            score = calc_quality(image, mask)
            # score = (score * 255).astype(np.uint8)
            cv2.imwrite(bmp_path.replace(".bmp", "_Q.jpg"), score)

    # test torch
    # tensor = torch.rand(size=(4, 1, 256, 256), device="cuda")
    # score = calc_quality_torch(tensor)
