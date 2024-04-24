#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 19:32
@Message: null
"""
import os
import random
import shutil

import numpy as np
import torch
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingLR,CosineAnnealingWarmRestarts,StepLR
import torch.nn as nn
from torchvision.models import resnet18
import matplotlib.pyplot as plt

from utils.util import generate_save_epoch

if __name__ == "__main__":
    print("demo")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    freeze = False
    print(freeze)
    flag = not freeze
    print(flag)
    # mask_name = "../dataset/isic2017/trainval/masks/ISIC_0012099_segmentation.png"
    # png = Image.open(mask_name)
    # png.save("png.png")
    # temp_png = np.array(png)
    # modify_png = np.zeros_like(temp_png)
    # modify_png[temp_png <= 127] = 0
    # modify_png[temp_png > 127] = 1
    # modify_png = modify_png*255
    # new_png = Image.fromarray(modify_png)
    # new_png.save("new_png.png")
    # # 创建示例张量，维度为 (1, 262144, 7)
    # temp_tp = torch.randn(2, 4)
    # print(temp_tp)
    #
    # # 对维度 0 和维度 1 进行求和
    # sum_result = torch.sum(temp_tp, dim=1)
    #
    # print("原始张量形状:", temp_tp.shape)
    # print("对维度 0 和 1 求和后的结果形状:", sum_result.shape)
    # print(sum_result)

    # raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

    # COLORS_CV = [(0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128),
    #              (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128),
    #              (64, 128, 128), (192, 128, 128), (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128),
    #              (128, 64, 12)]
    #
    #
    # def convert_color(color_cv):
    #     colors = []
    #     for i, c in enumerate(color_cv):
    #         colors.append(int((c / 128 * 255) if c <= 128 else c))
    #
    #     return colors[0], colors[1], colors[2]
    #
    # # 测试转换
    # # original_color = (128, 0, 0)
    # COLORS_PIL = [convert_color(original_color) for original_color in COLORS_CV]
    # print("Original Color:\n", COLORS_CV)
    # print(COLORS_PIL)

    # ===========================================
    # 从B导给的VOC 数据集中随机挑选指定数量的样本用于训练
    # ===========================================
    # random.seed(42)
    # src_seg_dir = r"C:\Users\40977\Desktop\DNN\MyCode\jcx\dataset\VOCdevkit\VOC2007\SegmentationClass"
    # src_img_dir = r"C:\Users\40977\Desktop\DNN\MyCode\jcx\dataset\VOCdevkit\VOC2007\JPEGImages"
    #
    # dst_img_dir = "data/VOCdevkit/JPEGImages"
    # dst_seg_dir = "data/VOCdevkit/SegmentationClass"
    #
    # seg_list = os.listdir(src_seg_dir)
    # random.shuffle(seg_list)
    # for i, seg_name in enumerate(seg_list):
    #     seg_prefix, seg_extension = os.path.splitext(seg_name)
    #     img_name = seg_prefix + '.jpg'
    #     if i < 2000:
    #         src_seg_path = os.path.join(src_seg_dir, seg_name)
    #         dst_seg_path = os.path.join(dst_seg_dir, seg_name)
    #         src_img_path = os.path.join(src_img_dir, img_name)
    #         dst_img_path = os.path.join(dst_img_dir, img_name)
    #         shutil.copy(src_seg_path, dst_seg_path)
    #         shutil.copy(src_img_path, dst_img_path)
    #     else:
    #         break
    #
    # print("done")

    # # 保存模型权重epoch
    # epochs = 150
    # epoch_list = generate_save_epoch(epochs)
    # print(epoch_list)
    # # epoch_list = generate_save_epoch(epochs, period=20)
    # # print(epoch_list)
    #
    # for epoch in range(epochs):
    #     if epoch in epoch_list:
    #         print(epoch, "eval model")

    # # 生成有序数组
    # arr = torch.arange(2 * 3 * 4)
    #
    # # 将数组reshape为维度为(2, 3, 4)
    # arr = arr.reshape(2, 3, 4)
    # print(arr)
    #
    # tp1 = torch.sum(arr, axis=[0,1])
    # print(tp1)
    # tp = torch.sum(arr, dim=(0, 1))
    # print(tp)
    # max_epoch = 20
    # iters = 200
    # model = resnet18(pretrained=False)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    # mode = 'cosineAnnWarm'
    # if mode == 'cosineAnn':
    #     scheduler = CosineAnnealingLR(optimizer, T_max=5, eta_min=0)
    # elif mode == 'cosineAnnWarm':
    #     scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max_epoch, T_mult=1)
    #     '''
    #     以T_0=5, T_mult=1为例:
    #     T_0:学习率第一次回到初始值的epoch位置.
    #     T_mult:这个控制了学习率回升的速度
    #         - 如果T_mult=1,则学习率在T_0,2*T_0,3*T_0,....,i*T_0,....处回到最大值(初始学习率)
    #             - 5,10,15,20,25,.......处回到最大值
    #         - 如果T_mult>1,则学习率在T_0,(1+T_mult)*T_0,(1+T_mult+T_mult**2)*T_0,.....,(1+T_mult+T_mult**2+...+T_0**i)*T0,处回到最大值
    #             - 5,15,35,75,155,.......处回到最大值
    #     example:
    #         T_0=5, T_mult=1
    #     '''
    # plt.figure()
    # cur_lr_list = []
    # for epoch in range(max_epoch):
    #     for batch in range(iters):
    #         '''
    #         这里scheduler.step(epoch + batch / iters)的理解如下,如果是一个epoch结束后再.step
    #         那么一个epoch内所有batch使用的都是同一个学习率,为了使得不同batch也使用不同的学习率
    #         则可以在这里进行.step
    #         '''
    #         # scheduler.step(epoch + batch / iters)
    #         optimizer.step()
    #     scheduler.step()
    #     cur_lr = optimizer.param_groups[-1]['lr']
    #     cur_lr_list.append(cur_lr)
    #     # print('cur_lr:', cur_lr)
    # temp_lsit = cur_lr_list[:-1]
    # x_list = list(range(len(temp_lsit)))
    # plt.plot(x_list, temp_lsit)
    # plt.show()
