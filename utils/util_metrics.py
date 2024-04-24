#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/26 17:08
@Message: null
"""
import csv
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# TODO: 这个函数有问题，batch size 越大 f1 score 分数越小
def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice系数
    # --------------------------------------------#
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    temp_tp = temp_target[..., :-1] * temp_inputs
    tp = torch.sum(temp_tp, dim=[0, 1])
    fp = torch.sum(temp_inputs, dim=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], dim=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score


def calculate_score(inputs, target, smooth=1e-5, threshold=0.5):
    """
    对预测结果计算相关指标，包括dice系数、mIoU、precision以及recall
    参考 https://github.com/kiharalab/ACC-UNet/blob/main/Experiments/utils.py
    :param inputs: 预测结果
    :param target: 对应GT
    :param smooth: 调节系数，防止分母为零
    :param threshold: 阈值
    :return:
    """
    n, c, h, w = inputs.size()  # (1,7,512,512)
    nt, ht, wt, ct = target.size()  # (1,512,512,8)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)  # (1,262144,7)
    temp_target = target.view(n, -1, ct)[..., :-1]    # (1,262144,8)

    temp_inputs = torch.gt(temp_inputs, threshold).float()   # (1,262144,7)
    temp_tp = temp_target * temp_inputs   # (1,262144,7)

    dice_coef, mIoU, precision, recall = 0.0, 0.0, 0.0, 0.0
    for bs in range(n):
        tp = torch.sum(temp_tp[bs], dim=0)     # (7) 预测正确的正样本（gt像素点）
        fp = torch.sum(temp_inputs[bs], dim=0) - tp
        fn = torch.sum(temp_target[bs], dim=0) - tp
        # 计算dice coefficient，和f1 score 一样？
        temp_dice = (2 * tp + smooth) / (2 * tp + fp + fn + smooth)
        dice_coef += torch.mean(temp_dice).item()
        temp_mIoU = (tp + smooth) / (tp + fn + fp + smooth)
        mIoU += torch.mean(temp_mIoU).item()

        temp_precision = (tp + smooth) / (tp + fp + smooth)
        precision += torch.mean(temp_precision).item()
        temp_recall = (tp + smooth) / (tp + fn + smooth)
        recall += torch.mean(temp_recall).item()

    return dice_coef/n, mIoU/n, precision/n, recall/n


# 设标签宽W，长H
def fast_hist(a, b, n):
    # --------------------------------------------------------------------------------#
    #   a是转化成一维数组的标签，形状(H×W,)；b是转化成一维数组的预测结果，形状(H×W,)
    # --------------------------------------------------------------------------------#
    k = (a >= 0) & (a < n)
    # --------------------------------------------------------------------------------#
    #   np.bincount计算了从0到n**2-1这n**2个数中每个数出现的次数，返回值形状(n, n)
    #   返回中，写对角线上的为分类正确的像素点
    # --------------------------------------------------------------------------------#
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)


def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)


def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)


def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)


def compute_mIoU(gt_dir, pred_dir, png_name_list, num_classes, name_classes=None, save_info=None):
    print('Num classes', num_classes)
    # -----------------------------------------#
    #   创建一个全是0的矩阵，是一个混淆矩阵
    # -----------------------------------------#
    hist = np.zeros((num_classes, num_classes))

    # ------------------------------------------------#
    #   获得验证集标签路径列表，方便直接读取
    #   获得验证集图像分割结果路径列表，方便直接读取
    # ------------------------------------------------#
    gt_imgs = [join(gt_dir, x + ".png") for x in png_name_list]
    pred_imgs = [join(pred_dir, x + ".png") for x in png_name_list]

    # ------------------------------------------------#
    #   读取每一个（图片-标签）对
    # ------------------------------------------------#
    for ind in range(len(gt_imgs)):
        # ------------------------------------------------#
        #   读取一张图像分割结果，转化成numpy数组
        # ------------------------------------------------#
        pred = np.array(Image.open(pred_imgs[ind]))
        # ------------------------------------------------#
        #   读取一张对应的标签，转化成numpy数组
        # ------------------------------------------------#
        label = np.array(Image.open(gt_imgs[ind]))

        # 如果图像分割结果与标签的大小不一样，这张图片就不计算
        if len(label.flatten()) != len(pred.flatten()):
            print(
                'Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(
                    len(label.flatten()), len(pred.flatten()), gt_imgs[ind],
                    pred_imgs[ind]))
            continue

        # ------------------------------------------------#
        #   对一张图片计算21×21的hist矩阵，并累加
        # ------------------------------------------------#
        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        # 每计算10张就输出一下目前已计算的图片中所有类别平均的mIoU值
        if name_classes is not None and ind > 0 and ind % 10 == 0:
            print('{:d} / {:d}: mIou-{:0.2f}%; mPA-{:0.2f}%; Accuracy-{:0.2f}%'.format(
                ind,
                len(gt_imgs),
                100 * np.nanmean(per_class_iu(hist)),
                100 * np.nanmean(per_class_PA_Recall(hist)),
                100 * per_Accuracy(hist)
            )
            )
    # ------------------------------------------------#
    #   计算所有验证集图片的逐类别mIoU值
    # ------------------------------------------------#
    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)
    # ------------------------------------------------#
    #   逐类别输出一下mIoU值
    # ------------------------------------------------#
    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) +
                  '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '%; Precision-' +
                  str(round(Precision[ind_class] * 100, 2)) + "%")
            if save_info is not None:
                with open(os.path.join(save_info, "epoch_mIoU.txt"), 'a') as f:
                    f.write('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) +
                            '; Recall (equal to the PA)-' + str(
                        round(PA_Recall[ind_class] * 100, 2)) + '%; Precision-' +
                            str(round(Precision[ind_class] * 100, 2)) + "%\n")

    # -----------------------------------------------------------------#
    #   在所有验证集图像上求所有类别平均的mIoU值，计算时忽略NaN值
    # -----------------------------------------------------------------#
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 2)) + '%; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 2)) + '%; Precision: ' + str(round(np.nanmean(Precision) * 100, 2)) + '%')
    return np.array(hist, np.int32), IoUs, PA_Recall, Precision
