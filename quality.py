import numpy as np
from scipy.ndimage import (
    correlate,
    generate_binary_structure,
    grey_dilation,
    grey_erosion,
    iterate_structure,
)

EPSILON = 1e-6


def calc_quality(image):
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
    kernel = generate_binary_structure(2, 1)
    e_kernel = iterate_structure(kernel, 7)
    d_kernel = iterate_structure(kernel, 11)
    opening = grey_erosion(image, structure=e_kernel)
    opening = grey_dilation(opening, structure=d_kernel)
    mask = np.where(opening < 150, 1, 0)

    # compute ocl
    ocl = np.where(eigv_max < EPSILON, EPSILON, 1.0 - eigv_min / (eigv_max + EPSILON))
    ocl = ocl * mask

    # compute sc
    sc = pow(eigv_max - eigv_min, 2) / pow(eigv_max + eigv_min + EPSILON, 2)
    sc = sc * mask

    # compute coh
    coh = (eigv_max - eigv_min) / (eigv_max + eigv_min + EPSILON)
    coh = coh * mask

    return mask, ocl, sc, coh


if __name__ == "__main__":
    from matplotlib.image import imread, imsave

    image = imread("images/train/tir/cjj-L1-1-first_1.bmp")
    mask, ocl, sc, coh = calc_quality(image)
    imsave("mask.jpg", mask * image, cmap="gray")
