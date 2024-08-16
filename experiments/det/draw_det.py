import glob
import os

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

# from readAndRoc import *
# from roc import *
from scipy.interpolate import interp1d
from scipy.optimize import brentq
from sklearn.metrics import auc, roc_curve

colors = list(mcolors.TABLEAU_COLORS.keys())


def showROC(model_name, colorname, y_label, y_pre, islog, isshow):
    # y_label = ([1, 1, 1, 2, 2, 2])  # 非二进制需要pos_label
    # y_label = ([0, 0, 0, 1, 1, 1])  # 1为正样本（分数高），0为负样本
    # y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
    # fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)

    fpr, tpr, thresholds = roc_curve(y_label, y_pre)

    for i, value in enumerate(thresholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label="{0} (AUC = {1:.3f})".format(model_name, roc_auc), lw=2, color=colorname)

    EER_x = [0.0001, 1]
    EER_y = [0.0001, 1]
    plt.plot(EER_x, EER_y, "r--")
    plt.xlim([0.0001, 1.0])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([0.0001, 1.0])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")  # 可以使用中文，但需要导入一些库即字体
    if islog:
        plt.xscale("log")
        # plt.yscale("log")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")  # fontsize=20
    if isshow:
        plt.show()


# def multi_models_roc(names, sampling_methods, colors, X_test, y_test, save=True, dpin=100):
#     """
#     将多个机器模型的roc图输出到一张图上
#
#     Args:
#         names: list, 多个模型的名称
#         sampling_methods: list, 多个模型的实例化对象
#         save: 选择是否将结果保存（默认为png格式）
#
#     Returns:
#         返回图片对象plt
#     """
#     plt.figure(figsize=(20, 20), dpi=dpin)
#
#     for (name, method, colorname) in zip(names, sampling_methods, colors):
#         method.fit(X_train, y_train)
#         y_test_preds = method.predict(X_test)
#         y_test_predprob = method.predict_proba(X_test)[:, 1]
#         fpr, tpr, thresholds = roc_curve(y_test, y_test_predprob, pos_label=1)
#
#         plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, auc(fpr, tpr)), color=colorname)
#         plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
#         plt.axis('square')
#         plt.xlim([0, 1])
#         plt.ylim([0, 1])
#         plt.xlabel('False Positive Rate', fontsize=20)
#         plt.ylabel('True Positive Rate', fontsize=20)
#         plt.title('ROC Curve', fontsize=25)
#         plt.legend(loc='lower right', fontsize=20)
#
#     if save:
#         plt.savefig('multi_models_roc.png')
#
#     return plt


def showDET_return_500_point(model_name, colorname, y_label, y_pre, islog, isshow, title):
    # y_label = ([1, 1, 1, 2, 2, 2])  # 非二进制需要pos_label
    # y_label = ([0, 0, 0, 1, 1, 1])  # 1为正样本（分数高），0为负样本
    # y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
    # fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)
    color = ["b", "g", "r", "c", "m", "y", "k", "w"]
    # plt.xscale("log")
    # plt.yscale("log")
    # plt.grid(True)
    for i in range(len(title)):
        model_name = title[i]
        colorname = color[i]
        y_p = y_pre[i]
        fpr, tpr, thresholds = roc_curve(y_label, y_p)
        eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
        thresh = interp1d(fpr, thresholds)(eer)

        # for i, value in enumerate(thresholds):
        #     print("%f %f %f" % (fpr[i], tpr[i], value))

        # roc_auc = auc(fpr, tpr)
        frr = 1 - tpr

        plt.plot(fpr, frr, label="{0} (EER = {1:.5f})".format(model_name, eer), lw=1, color=colorname)

        EER_x = [0, 1]
        EER_y = [0, 1]
        plt.plot(EER_x, EER_y, "r--")
        plt.xlim([0, 0.005])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
        plt.ylim([0, 0.005])
        plt.xlabel("False Non-Match Rate")
        plt.ylabel("False Match Rate")  # 可以使用中文，但需要导入一些库即字体

    if islog:
        plt.yscale("log")
        plt.grid(True)
        plt.grid(True)
    plt.title("DET Curve")
    plt.legend(loc="upper right")  # fontsize=20
    if isshow:
        plt.show()
    else:
        plt.savefig("DET_curve.png")
    # func = interp1d(fpr, frr)
    # x_new = np.logspace(start=-5, stop=0, num=500)
    # y_new = func(x_new)
    # return y_new, x_new


def showDET(model_name, colorname, y_label, y_pre, islog, isshow):
    # y_label = ([1, 1, 1, 2, 2, 2])  # 非二进制需要pos_label
    # y_label = ([0, 0, 0, 1, 1, 1])  # 1为正样本（分数高），0为负样本
    # y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
    # fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)

    fpr, tpr, thresholds = roc_curve(y_label, y_pre)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    for i, value in enumerate(thresholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    # roc_auc = auc(fpr, tpr)
    frr = 1 - tpr

    plt.plot(fpr, frr, label="{0} (EER = {1:.3f})".format(model_name, eer), lw=2, color=colorname)

    EER_x = [0.0001, 1]
    EER_y = [0.0001, 1]
    plt.plot(EER_x, EER_y, "r--")
    plt.xlim([0.0001, 1.0])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([0.0001, 1.0])
    plt.xlabel("False Non-Match Rate")
    plt.ylabel("False Match Rate")  # 可以使用中文，但需要导入一些库即字体
    if islog:
        plt.xscale("log")
        plt.yscale("log")
    plt.title("DET Curve")
    plt.legend(loc="lower right")  # fontsize=20
    if isshow:
        plt.show()


def cac_DET(model_name, colorname, x_in, y_in, islog, isshow):
    # y_label = ([1, 1, 1, 2, 2, 2])  # 非二进制需要pos_label
    # y_label = ([0, 0, 0, 1, 1, 1])  # 1为正样本（分数高），0为负样本
    # y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])
    # fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=2)
    fpr, frr = x_in, y_in
    eer = brentq(lambda x: x - interp1d(fpr, frr)(x), 0.0001, 1.0)

    roc_auc = auc(fpr, 1 - frr)
    # frr = 1 - tpr
    # np.save('/home/student/student3/mycode/python_code/pydataset/roc_np/DG_capability/' + model_name + '.npy', [x_in, y_in])
    # np.save('/home/student/student3/mycode/python_code/pydataset/peak_rate/' + model_name + '.npy',
    #         [x_in, y_in])

    plt.plot(fpr, frr, label="{0} (EER = {1:.7f}, AUC = {2:.7f})".format(model_name, eer, roc_auc), lw=2)

    EER_x = [0.0001, 1]
    EER_y = [0.0001, 1]
    plt.plot(EER_x, EER_y, "r--")
    plt.xlim([0.0001, 1.0])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([0.0001, 1.0])
    plt.xlabel("False Non-Match Rate")
    plt.ylabel("False Match Rate")  # 可以使用中文，但需要导入一些库即字体
    if islog:
        plt.xscale("log")
        plt.yscale("log")
    plt.title("DET Curve")
    plt.legend(loc="lower right")  # fontsize=20
    if isshow:
        plt.show()


def DET_return_500_point(y_label, y_pre):
    fpr, tpr, thresholds = roc_curve(y_label, y_pre)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    frr = 1 - tpr

    func = interp1d(fpr, frr)
    x_new = np.logspace(start=-5, stop=0, num=500)
    y_new = func(x_new)
    return y_new, x_new


def readDataFromPath(path):
    files = glob.glob(path + "*.txt")
    lists = []
    lists.append([])
    lists.append([])
    lists.append([])
    lists.append([])
    for file in files:
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
    return lists


# 加载预测值与标签值 这个是为了绘制BScan的DET图的加载文件
def loadtxtBScan():
    benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0AdatasetAAECover40_P_AECover40P_35_all/"
    defectPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0AdefectAECover40_P_AECover40P_35_all/"
    spoofPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0AspoofAECover40_P_AECover40P_35_all/"
    listBenefit = readDataFromPath(benefitPath)
    listDefect = readDataFromPath(defectPath)
    listSpoot = readDataFromPath(spoofPath)
    listLabel = len(listBenefit[1]) * [0] + (len(listDefect[1]) + len(listSpoot[1])) * [1]
    listPre = listBenefit[1] + listDefect[1] + listSpoot[1]
    thr = 3000  # 2500
    listPre = (np.clip(listPre, 0, thr) / thr).tolist()
    x, y = showDET_return_500_point(model_name="chapter3", colorname=mcolors.TABLEAU_COLORS[colors[2]], y_label=listLabel, y_pre=listPre, islog=0, isshow=1)
    drawROC22(listPre, listLabel, "bScan", "bScan")


#


def getlabelaAndScope():
    # benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BdatasetAAECover40_P_AECover40P_35_all/"
    benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BspoofAECover40_P_AECover40P_35_all/"
    num = 0
    total = 0
    files = glob.glob(benefitPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        listBenefit = (np.clip(lists[1], 0, thr) / thr).tolist()
        listBenefit = np.array(listBenefit[::7])

        scope = listBenefit[80:120].mean()

        print(txtname, "-------", scope)
    print("符合要求的有：", num, "个")
    print("一同有", total)


def IntanceUse_spoofpercent():
    caiyang = 7
    benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BdatasetAAECover40_P_AECover40P_35_all/"
    # benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BspoofAECover40_P_AECover40P_35_all/"
    num = 0
    total = 0
    scopes = []
    files = glob.glob(benefitPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        listBenefit = (np.clip(lists[1], 0, thr) / thr).tolist()
        listBenefit = np.array(listBenefit[::caiyang])

        # scope = sum(1 for num in listBenefit if num > 0.013198789209127426) / 200
        # scopes.append(scope)

    # print("符合要求的有：", num, "个")
    benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BspoofAECover40_P_AECover40P_35_all/"
    files = glob.glob(benefitPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 256 * 256  # 2500
        listBenefit = (np.clip(lists[1], 0, thr) / thr).tolist()
        listBenefit = np.array(listBenefit[::caiyang])

        # scope = sum(1 for num in listBenefit if num > 0.013198789209127426) / 200
        # scopes.append(scope)

    print(total)
    drawHist(scopes[:120], scopes[120:], np.arange(0, 1.02, 0.01))
    listlabel = np.array(120 * [0] + 105 * [1])
    drawROC22(scopes, listlabel, "charpetINS", tittle="charpetINS")
    # x,y = showDET_return_500_point(model_name="chapter3",colorname=mcolors.TABLEAU_COLORS[colors[2]],y_label=listlabel,y_pre=scopes,islog=0,isshow=1)
    showDET("charpetINS", "r", listlabel, scopes, 1, 1)


def justDET():
    num = 0
    total = 0
    scopes = []
    title = []
    benefitPath = "data1/det_score/"
    labelPath = "data1/data1_label.txt"
    files = glob.glob(benefitPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0].split("_")[1]
        lists = []
        # lists.append([])
        # lists.append([])
        # lists.append([])
        # lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")[:15553]
            contents = list(map(int, contents))
            # i = 0
            # for x in contents:
            #     print("i",i)
            #     lists[i] = lists[i] + (eval(x))
            #     i = i + 1
        # thr = 3000  # 2500
        # listBenefit = (np.clip(lists[1], 0, thr) / thr).tolist()
        # listBenefit = np.array(listBenefit[::7])
        #
        # scope = sum(1 for num in listBenefit if num > 0.2) / 200
        # scopes.append(scope)
        #
        # if scope > 0.35:
        #     num = num + 1
        # print(txtname, "-------", scope)
        scopes.append(contents)
        title.append(txtname)
    print(total)
    # with open(benefitPath, "r") as f:
    #     content = f.read()
    #     label = content.split('\n')[:4875]
    #     scopes = list(map(int, label))
    # scopes1 = scopes[1]
    with open(labelPath, "r") as f:
        content = f.read()
        label = content.split("\n")[:15553]
        label = list(map(int, label))
    LABEL = []
    for i in range(7):
        LABEL.append(label)
    # label2 = np.array(4875*[1])
    showDET_return_500_point(
        model_name="chapter3", colorname=mcolors.TABLEAU_COLORS[colors[2]], y_label=label, y_pre=scopes, islog=False, isshow=True, title=title
    )


def InstanceValuation():
    caiyang = 7
    benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BdatasetAAECover40_P_AECover40P_35_all/"
    # benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BspoofAECover40_P_AECover40P_35_all/"
    num = 0
    total = 0
    scopes = []
    files = glob.glob(benefitPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        listBenefit = (np.clip(lists[1], 0, thr) / thr).tolist()
        listBenefit = np.array(listBenefit[::caiyang])
        # todo 这里添加 体数据的分数的计算方式
        scope = getInstanceScopeByMean(listBenefit)
        # scope = getInstanceScopeByMean(lists[2])
        scopes.append(scope)

    # print("符合要求的有：", num, "个")
    spoofPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BspoofAECover40_P_AECover40P_35_all/"
    files = glob.glob(spoofPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        listspoof = (np.clip(lists[1], 0, thr) / thr).tolist()
        listspoof = np.array(listspoof[::caiyang])
        scope = getInstanceScopeByMean(listspoof)
        # scope = getInstanceScopeByMean(lists[2])
        scopes.append(scope)

    # drawHist(scopes[:120], scopes[120:], np.arange(0, 1.02, 0.01))
    listlabel = np.array(120 * [0] + 105 * [1])
    # drawROC22(scopes, listlabel, 'charpetINS', tittle='charpetINS')
    return scopes, listlabel
    x, y = showDET_return_500_point(model_name="chapter3", colorname=mcolors.TABLEAU_COLORS[colors[2]], y_label=listlabel, y_pre=scopes, islog=0, isshow=1)


# 计算体数据得分的策略 通过不同位置的得分计算 ×权重 作为体数据的得分
#  InstanceList
def getInstanceScopeByWeight(InstanceList):

    print(InstanceList)


# 使用均值
#  结果：错误率提高 0.14
def getInstanceScopeByMeanrooer(InstanceList):
    return np.array(InstanceList).mean()


# 使用加权的均值
def getInstanceScopeByMean(InstanceList):
    ll = int(len(InstanceList) / 5)
    before = np.array(InstanceList[ll : ll * 2]).mean()
    after = np.array(InstanceList[ll * 3 : ll * 4]).mean()
    mid = np.array(InstanceList[ll * 2 : ll * 3]).mean()
    scope1 = (mid * 2 + (before + after) * 0.5) / 3
    scope2 = (before * 2 + (mid + after) * 0.5) / 3
    scope3 = (after * 2 + (before + mid) * 0.5) / 3

    return min(scope1, scope2, scope3)
    # return mid


def getInstanceAUCfprtpr():
    anomaly_score_prediction, anomaly_score_gt = InstanceValuation()
    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)

    fpr, tpr, thresholds = roc_curve(anomaly_score_gt, anomaly_score_prediction)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    print(eer)
    return fpr, tpr


def getSingleInstanceAucfprtpr():
    caiyang = 14
    benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/datasetAAECoverSingle_P_oldvision_AECoverSingle_P_32_32_1400/"
    # benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BspoofAECover40_P_AECover40P_35_all/"
    num = 0
    total = 0
    scopes = []
    files = glob.glob(benefitPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        listBenefit = (np.clip(lists[1], 0, thr) / thr).tolist()
        listBenefit = np.array(listBenefit[::caiyang])
        # todo 这里添加 体数据的分数的计算方式
        scope = getInstanceScopeByMean(listBenefit)
        scopes.append(scope)

    # print("符合要求的有：", num, "个")
    spoofPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/spoofAECoverSingle_P_oldvision_AECoverSingle_P_32_32_1400/"
    files = glob.glob(spoofPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        listspoof = (np.clip(lists[1], 0, thr) / thr).tolist()
        listspoof = np.array(listspoof[::caiyang])
        scope = getInstanceScopeByMean(listspoof)
        scopes.append(scope)

    # drawHist(scopes[:120], scopes[120:], np.arange(0, 1.02, 0.01))
    listlabel = np.array(120 * [0] + 105 * [1])
    # drawROC22(scopes, listlabel, 'charpeSingletINS', tittle='charpeSingletINS')
    anomaly_score_prediction, anomaly_score_gt = scopes, listlabel
    anomaly_score_prediction = np.array(anomaly_score_prediction)
    anomaly_score_gt = np.array(anomaly_score_gt)

    fpr, tpr, thresholds = roc_curve(anomaly_score_gt, anomaly_score_prediction)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    return fpr, tpr
    # x,y = showDET_return_500_point(model_name="chapter3",colorname=mcolors.TABLEAU_COLORS[colors[2]],y_label=listlabel,y_pre=scopes,islog=0,isshow=1)


def InstanceValuation2():
    benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/datasetAAECover40_P_dulForPart/"
    # benefitPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/0BspoofAECover40_P_AECover40P_35_all/"
    num = 0
    total = 0
    scopes = []
    files = glob.glob(benefitPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        # listBenefit = (np.clip(lists[2], 0, thr) / thr).tolist()
        listBenefit = lists[2]
        listBenefit = np.array(listBenefit[::7])
        # todo 这里添加 体数据的分数的计算方式
        scope = getInstanceScopeByMean(listBenefit)
        # scope = listBenefit.mean()
        scopes.append(scope)

    # print("符合要求的有：", num, "个")
    spoofPath = "/home/student/student4/Project_xk/Chapter3/new_rec_out_test/spoofAECover40_P_dulForPart/"
    files = glob.glob(spoofPath + "*.txt")
    for file in files:
        total = total + 1
        txtname = os.path.basename(file).split(".")[0]
        lists = []
        lists.append([])
        lists.append([])
        lists.append([])
        lists.append([])
        with open(file, "r") as f:
            content = f.read()
            contents = content.split("\n")
            i = 0
            for x in contents:
                # print("i",i)
                lists[i] = lists[i] + (eval(x))
                i = i + 1
        thr = 3000  # 2500
        # listspoof = (np.clip(lists[1], 0, thr) / thr).tolist()
        listspoof = lists[2]
        listspoof = np.array(listspoof[::7])
        scope = getInstanceScopeByMean(listspoof)
        # scope = listspoof.mean()
        scopes.append(scope)

    # drawHist(scopes[:120], scopes[120:], np.arange(0, 1.02, 0.01))
    listlabel = np.array(640 * [0] + 525 * [1])
    drawROC22(scopes, listlabel, "charpetINS", tittle="charpetINS")
    # return scopes,listlabel
    x, y = showDET_return_500_point(model_name="chapter3", colorname=mcolors.TABLEAU_COLORS[colors[2]], y_label=listlabel, y_pre=scopes, islog=0, isshow=1)
    drawHist(scopes[:640], scopes[640:], np.arange(0, 10, 0.005), xlim=0.3)
    # drawHist(listsTrans[:num_real], listsTrans[num_real:], np.arange(0, 3, 0.0001), xlim=0.05)


if __name__ == "__main__":
    justDET()
    # getInstanceAUCfprtpr()
    # getSingleInstanceAucfprtpr
    # getSingleInstanceAucfprtpr()
    # InstanceValuation()
    # InstanceValuation()
    # getSingleInstanceAucfprtpr()
    # getSingleInstanceAucfprtpr()
    # getlabelaAndScope()
    # getInstanceAUCfprtpr()
    # IntanceUse_spoofpercent()
