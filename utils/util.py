#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 19:39
@Message: null
"""
import math
import os
import random
import numpy as np

import torch
from PIL import Image, ImageDraw, ImageFont
from matplotlib import pyplot as plt

# CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
#            "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# CLASSES_NAME = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
#                 "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# CLASSES = ["background", "H", "T1", "T2", "T3", "T4", "T5"]
# CLASSES_NAME = ["H", "T1", "T2", "T3", "T4", "T5"]

CLASSES = ["background", "H", "T1", "T2"]
CLASSES_NAME = ["H", "T1", "T2"]

COLORS_CV = [
    (0, 0, 0), (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128), (0, 128, 128), (128, 128, 128),
    (64, 0, 0), (192, 0, 0), (64, 128, 0), (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128), (128, 64, 12)
]

COLORS_PIL = [
    (255, 0, 0), (0, 255, 0), (255, 255, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255), (255, 255, 255), (127, 0, 0),
    (192, 0, 0), (127, 255, 0), (192, 255, 0), (127, 0, 255), (192, 0, 255), (127, 255, 255), (192, 255, 255),
    (0, 127, 0), (255, 127, 0), (0, 192, 0), (255, 192, 0), (0, 127, 255), (255, 127, 23)
]


def set_random_seed(seed=42):
    """
    训练中的随机种子，保证相同参数结果可复现
    :param seed: 随机种子
    :return:
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    print('You have chosen to seed training. This will slow down your training!')


def convert_color(color_cv):
    colors = []
    for i, c in enumerate(color_cv):
        colors.append(int((c / 128 * 255) if c <= 128 else c))

    return colors[0], colors[1], colors[2]


def cvtColor(image):
    """
    将图像转换成RGB图像，防止灰度图在预测时报错。
    代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
    :param image: PIL 类型的图像对象
    :return: 转换为RGB格式的 PIL 类型的图像对象
    """
    if len(np.shape(image)) == 3 and np.shape(image)[2] == 3:
        return image
    else:
        image = image.convert('RGB')
        return image


def resize_image(image,
                 size: tuple = (512, 512)):
    """
    对输入图像进行不失真 resize
    :param image: PIL 类型的图像对象
    :param size: 目标尺寸
    :return: 调整尺寸后的图片，添加的灰度条的宽高
    """
    iw, ih = image.size
    w, h = size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)

    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128, 128, 128))
    new_image.paste(image, ((w - nw) // 2, (h - nh) // 2))

    return new_image, nw, nh


def preprocess_input(image):
    """
    数据标准化？
    :param image:
    :return:
    """
    image /= 255.0
    return image


def generate_dir(dir_path):
    """
    若目标文件夹不存在，则自动创建
    :param dir_path: 需要创建的文件夹路径
    :return:
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def generate_save_epoch(total_epoch: int, period: int = -1):
    """
    基于二分法或固定间隔次数生成需要进行评估的轮次列表
    :param total_epoch: 本次训练的轮次数
    :param period: 间隔次数
    :return: 生成的评估轮次列表
    """
    target_list = []
    if period == -1:
        target = 0
        remainder = total_epoch
        while target < total_epoch - 1:
            target = target + remainder // 2 + 1
            target_list.append(target - 1)
            remainder = total_epoch - target
    else:
        for epoch in range(total_epoch):
            if (epoch + 1) % period == 0:
                target_list.append(epoch)
    return target_list


def draw_filled_rectangle(draw,
                          rectangle_position: tuple = (50, 50),
                          rectangle_size: int = 100,
                          rectangle_color: tuple = (255, 0, 0)):
    """
        输入 ImageDraw 对象绘制一个填充指定颜色的矩形并返回结果。
    :param draw: ImageDraw.Draw(image)
    :param rectangle_position: 矩形位置
    :param rectangle_size: 矩形大小
    :param rectangle_color: 矩形填充颜色
    :return: ImageDraw
    """
    # 矩形的左上角坐标
    x, y = rectangle_position
    width = rectangle_size
    height = rectangle_size

    # 计算矩形的右下角坐标
    x1, y1 = x + width * 1.5, y + height

    # 绘制填充的彩色矩形
    draw.rectangle([rectangle_position, (x1, y1)], fill=rectangle_color)
    return draw


def add_info_to_image(image,
                      scale_factor: int = 23,
                      font_color: tuple = (0, 0, 0),
                      colors=None,
                      name_classes=None):
    """
        向图像中添加文本信息并返回处理好的图像
    :param image: Image
    :param scale_factor: 文本相对于图像尺寸计算得到的缩放因子
    :param font_color: 文本颜色，PIL 颜色编码
    :param colors: 矩形填充颜色，默认即可
    :param name_classes: 类别名称，默认即可
    :return: Image
    """
    if colors is None:
        colors = COLORS_PIL
    if name_classes is None:
        name_classes = CLASSES_NAME

    w, h = image.size
    min_size = min(w, h)
    unit = min_size // scale_factor
    # distance = (h - unit * 2) // len(name_classes)
    distance = unit
    # 自适应字体大小
    font_size = unit // 1.2
    # 字体位置：底部
    x, y = (unit//3, unit)
    draw = ImageDraw.Draw(image)

    font = ImageFont.truetype(font="utils/arial.ttf", size=font_size, index=0)

    for i in range(len(name_classes)):
        text = " " + name_classes[i]
        position = (x, y)
        position_text = (x + unit * 1.5 + 4, y + unit//8)
        rectangle_size = unit
        draw = draw_filled_rectangle(draw, position, rectangle_size, rectangle_color=colors[i])
        draw.text(position_text, text, font=font, fill=font_color)
        y = y + distance + 4

    return image


def save_info_when_training(train_loss, val_loss, lr_list, save_path='', save=True):
    X = range(0, len(train_loss))
    y_train = train_loss
    plt.figure(figsize=(9, 5))
    plt.subplot(1, 2, 1)
    plt.plot(X, lr_list, color='green', linestyle='-', marker='.', label='lr')
    plt.title('lr vs epochs item')
    plt.ylabel('Lr')
    plt.xlabel('Epochs')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
    plt.legend()
    plt.grid(True)

    x_loss = range(0, len(val_loss))
    plt.subplot(1, 2, 2)
    plt.plot(x_loss, y_train, 'red', linestyle='-', marker='.', label='train')
    plt.title('train/val loss vs epochs item')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    y_val = val_loss
    plt.plot(X, y_val, 'SteelBlue', linestyle='-', marker='.', label='val')
    plt.legend()
    plt.grid(True)

    # x_acc = range(0, len(val_acc))
    # plt.subplot(1, 3, 3)
    # plt.plot(x_acc, val_acc, 'g.-', label='val')
    # plt.title('val f_score vs epochs item')
    # plt.ylabel('val f_score')
    # plt.grid(True)
    # # 设置 Y 轴范围从 0 到最大值
    # max_value = max(val_acc) * 1.1
    # plt.ylim(0, max_value)
    plt.tight_layout()
    if save:
        plt.savefig(save_path + '/epoch_loss.png')
    else:
        plt.show()
    plt.cla()
    plt.close("all")


def set_optimizer_lr(optimizer, current_epoch, max_epochs, warmup_epochs=10, max_lr=0.1, min_lr=0.000001, lr_type="warmup_cosine"):
    if lr_type == "warmup_cosine":
        if current_epoch < warmup_epochs:
            lr = max_lr * current_epoch / warmup_epochs + min_lr
        else:
            lr = min_lr + (max_lr - min_lr) * (
                        1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (max_epochs - warmup_epochs))) / 2
    else:
        raise ValueError("Unsupported lr type for {}".format(lr_type))

    # 基于计算出的学习率进行赋值
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
