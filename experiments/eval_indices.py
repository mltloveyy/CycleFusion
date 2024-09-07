import math
import os

import cv2
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim


#### 基于信息论的评估指标 ####
def EN_function(F):
    histogram, _ = np.histogram(F, bins=256, range=(0, 255))
    histogram = histogram / float(np.sum(histogram))
    EN = -np.sum(histogram * np.log2(histogram + 1e-7))
    return EN


def MI_function(A, B, F, gray_level=256):
    def Hab(a, b, gray_level):
        hang, lie = a.shape
        N = gray_level
        h = np.zeros((N, N))
        for i in range(hang):
            for j in range(lie):
                h[a[i, j], b[i, j]] = h[a[i, j], b[i, j]] + 1
        h = h / np.sum(h)
        a_marg = np.sum(h, axis=0)
        b_marg = np.sum(h, axis=1)
        H_x = 0
        H_y = 0
        for i in range(N):
            if a_marg[i] != 0:
                H_x = H_x + a_marg[i] * math.log2(a_marg[i])
        for i in range(N):
            if b_marg[i] != 0:
                H_x = H_x + b_marg[i] * math.log2(b_marg[i])
        H_xy = 0
        for i in range(N):
            for j in range(N):
                if h[i, j] != 0:
                    H_xy = H_xy + h[i, j] * math.log2(h[i, j])
        r = H_xy - H_x - H_y
        return r

    MIA = Hab(A, F, gray_level)
    MIB = Hab(B, F, gray_level)
    MI = 0.5 * MIA + 0.5 * MIB
    return MI


def PSNR_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    PSNR = 20 * np.log10(255 / np.sqrt(MSE))
    return PSNR


#### 基于结构相似性的评估指标 ####
def SSIM_function(A, B, F):
    ssimAF = ssim(A, F, data_range=255, win_size=11, multichannel=True)
    ssimBF = ssim(B, F, data_range=255, win_size=11, multichannel=True)
    SSIM = 0.5 * ssimAF + 0.5 * ssimBF
    return SSIM


def MS_SSIM_function(A, B, F):
    def average_pool(img, factor):
        return img.reshape(img.shape[0] // factor, factor, img.shape[1] // factor, factor).mean(axis=(1, 3))

    def resize_image(img, factor):
        new_h = img.shape[0] // factor * factor
        new_w = img.shape[1] // factor * factor
        return cv2.resize(img.astype(np.float32), (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    def ms_ssim(a, b, max_val=255.0, levels=5, win_sizes=[11, 11, 11, 11, 11], weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333]):
        mssim = []
        a = resize_image(a, 2**levels)
        b = resize_image(b, 2**levels)

        for i in range(levels):
            if i == 0:
                a_down = a
                b_down = b
            else:
                a_down = average_pool(a, 2**i)
                b_down = average_pool(b, 2**i)

            ssim_val = ssim(a_down, b_down, data_range=max_val, win_size=win_sizes[0], multichannel=True)
            mssim.append(ssim_val ** weights[i])

        ms_ssim_val = np.prod(mssim) ** (1.0 / np.sum(weights))

        return ms_ssim_val

    ms_ssimAF = ms_ssim(A, F)
    ms_ssimBF = ms_ssim(B, F)
    MS_SSIM = 0.5 * ms_ssimAF + 0.5 * ms_ssimBF
    return MS_SSIM


def MSE_function(A, B, F):
    A = A / 255.0
    B = B / 255.0
    F = F / 255.0
    m, n = F.shape
    MSE_AF = np.sum(np.sum((F - A) ** 2)) / (m * n)
    MSE_BF = np.sum(np.sum((F - B) ** 2)) / (m * n)
    MSE = 0.5 * MSE_AF + 0.5 * MSE_BF
    return MSE


#### 基于图像特征的评估指标 ####
def SF_function(F):
    RF = np.diff(F, axis=0)
    RF1 = np.sqrt(np.mean(np.mean(RF**2)))
    CF = np.diff(F, axis=1)
    CF1 = np.sqrt(np.mean(np.mean(CF**2)))
    SF = np.sqrt(RF1**2 + CF1**2)
    return SF


def SD_function(F):
    m, n = F.shape
    u = np.mean(F)
    SD = np.sqrt(np.sum(np.sum((F - u) ** 2)) / (m * n))
    return SD


def AG_function(F):
    width = F.shape[1]
    width = width - 1
    height = F.shape[0]
    height = height - 1
    [grady, gradx] = np.gradient(F)
    s = np.sqrt((np.square(gradx) + np.square(grady)) / 2)
    AG = np.sum(np.sum(s)) / (width * height)
    return AG


#### 基于人类视觉感知的评估指标 ####
def VIF_function(A, B, F):
    def fspecial_gaussian(shape, sigma):
        m, n = [(ss - 1.0) / 2.0 for ss in shape]
        y, x = np.ogrid[-m : m + 1, -n : n + 1]
        h = np.exp(-(x * x + y * y) / (2.0 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    def vifp_mscale(ref, dist):
        sigma_nsq = 2
        num = 0
        den = 0
        for scale in range(1, 5):
            N = 2 ** (4 - scale + 1) + 1
            win = fspecial_gaussian((N, N), N / 5)

            if scale > 1:
                ref = convolve2d(ref, win, mode="valid")
                dist = convolve2d(dist, win, mode="valid")
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, win, mode="valid")
            mu2 = convolve2d(dist, win, mode="valid")
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, win, mode="valid") - mu1_sq
            sigma2_sq = convolve2d(dist * dist, win, mode="valid") - mu2_sq
            sigma12 = convolve2d(ref * dist, win, mode="valid") - mu1_mu2
            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + 1e-10)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < 1e-10] = 0
            sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
            sigma1_sq[sigma1_sq < 1e-10] = 0

            g[sigma2_sq < 1e-10] = 0
            sv_sq[sigma2_sq < 1e-10] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= 1e-10] = 1e-10

            num += np.sum(np.log10(1 + g**2 * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
        vifp = num / den
        return vifp

    VIF = vifp_mscale(A, F) + vifp_mscale(B, F)
    return VIF


#### 基于源图像与生成图像的评估指标 ####
def CC_function(A, B, F):
    rAF = np.sum((A - np.mean(A)) * (F - np.mean(F))) / np.sqrt(np.sum((A - np.mean(A)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    rBF = np.sum((B - np.mean(B)) * (F - np.mean(F))) / np.sqrt(np.sum((B - np.mean(B)) ** 2) * np.sum((F - np.mean(F)) ** 2))
    CC = np.mean([rAF, rBF])
    return CC


def SCD_function(A, B, F):
    def corr2(a, b):
        a = a - np.mean(a)
        b = b - np.mean(b)
        r = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
        return r

    SCD = corr2(F - B, A) + corr2(F - A, B)
    return SCD


def Qabf_function(A, B, F):
    edgesA = sobel(A)
    edgesB = sobel(B)
    edgesF = sobel(F)
    ssimAF = ssim(edgesA, edgesF, data_range=255, win_size=11, multichannel=True)
    ssimBF = ssim(edgesB, edgesF, data_range=255, win_size=11, multichannel=True)
    Qabf = 0.5 * ssimAF + 0.5 * ssimBF
    return Qabf


def Nabf_function(A, B, F):
    diffAF = np.abs(A - F)
    diffBF = np.abs(B - F)
    Nabf = 0.5 * np.std(diffAF) + 0.5 * np.std(diffBF)
    return Nabf


def eval_indices(ir_name, vi_name, f_name):
    ir_img = cv2.imread(ir_name, cv2.IMREAD_UNCHANGED)
    vi_img = cv2.imread(vi_name, cv2.IMREAD_UNCHANGED)
    f_img = cv2.imread(f_name, cv2.IMREAD_UNCHANGED)

    ir_img_int = ir_img.astype(np.int32)
    ir_img_double = ir_img.astype(np.float32)

    vi_img_int = vi_img.astype(np.int32)
    vi_img_double = vi_img.astype(np.float32)

    f_img_int = f_img.astype(np.int32)
    f_img_double = f_img.astype(np.float32)

    results = {}

    results["EN"] = EN_function(f_img_int)
    results["MI"] = MI_function(ir_img_int, vi_img_int, f_img_int, gray_level=256)
    results["PSNR"] = PSNR_function(ir_img_double, vi_img_double, f_img_double)

    results["SSIM"] = SSIM_function(ir_img_int, vi_img_int, f_img_int)
    results["MS_SSIM"] = MS_SSIM_function(ir_img_int, vi_img_int, f_img_int)
    results["MSE"] = MSE_function(ir_img_double, vi_img_double, f_img_double)

    results["SF"] = SF_function(f_img_double)
    results["SD"] = SD_function(f_img_double)
    results["AG"] = AG_function(f_img_double)

    results["VIF"] = VIF_function(ir_img_double, vi_img_double, f_img_double)

    results["CC"] = CC_function(ir_img_double, vi_img_double, f_img_double)
    results["SCD"] = SCD_function(ir_img_double, vi_img_double, f_img_double)
    results["Qabf"] = Qabf_function(ir_img_int, vi_img_int, f_img_int)
    results["Nabf"] = Nabf_function(ir_img_double, vi_img_double, f_img_double)

    return results


if __name__ == "__main__":
    tir_path = r"experiments\results\1208\tir"
    oct_path = r"experiments\results\1208\oct"
    fused_path_list = [
        # r"experiments\results\1208\1208_ours",
        # r"experiments\results\1208\1208_cnn",
        # r"experiments\results\1208\csmca",
        r"experiments\results\1208\darlow1208",
        r"experiments\results\1208\DTNP-MIF1208",
    ]
    method_list = [
        # "proposed",
        # "densefuse",
        # "SHD",
        "darlow",
        "DTNP_MIF",
    ]
    index_list = ["EN", "MI", "PSNR", "SSIM", "MS_SSIM", "MSE", "SF", "SD", "AG", "VIF", "CC", "SCD", "Qabf", "Nabf"]

    summary = []
    for i, fused_path in enumerate(fused_path_list):
        method = method_list[i]
        print(f"Starting {method}")
        column_sums = {col: 0 for col in index_list}
        row_count = 0
        for name in os.listdir(fused_path):
            print(f"Processing No.{row_count}")
            tir_name = os.path.join(tir_path, name)
            oct_name = os.path.join(oct_path, name)
            fused_name = os.path.join(fused_path, name)
            results = eval_indices(tir_name, oct_name, fused_name)
            for col in index_list:
                column_sums[col] += results.get(col, 0)
            row_count += 1
        mean_indices = {col: sum_ / row_count for col, sum_ in column_sums.items()}
        mean_indices["method"] = method
        summary.append(mean_indices)
    summary = pd.DataFrame(summary, columns=["method", *index_list])
    summary.to_excel("summary.xlsx", index=False)
