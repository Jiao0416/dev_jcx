#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 20:45
@Message: null
"""

import os
import random

import numpy as np
from PIL import Image
from tqdm import tqdm

from utils.util import generate_dir

if __name__ == "__main__":
    print("voc_annotation")

    # 训练集/验证集占所有数据比例，此处不单独划分测试集，想要增加测试集修改train_val_percent
    train_val_percent = 1

    # 修改train_percent用于改变验证集的比例 9:1
    train_percent = 0.8

    # VOC数据集所在的文件夹
    VOCdevkit_path = '../data/VOCdevkit'
    mask_image_path = 'SegmentationClass'
    imagesets_path = 'ImageSets/Segmentation'

    random.seed(0)
    print("Generate txt in ImageSets.")
    segfilepath = os.path.join(VOCdevkit_path, mask_image_path)
    saveBasePath = os.path.join(VOCdevkit_path, imagesets_path)
    # 若文件夹不存在自动创建
    generate_dir(saveBasePath)
    generate_dir(segfilepath)

    temp_seg = os.listdir(segfilepath)
    total_seg = []
    for seg in temp_seg:
        if seg.endswith(".png"):
            total_seg.append(seg)

    num = len(total_seg)
    num_list = range(num)
    tv = int(num * train_val_percent)
    tr = int(tv * train_percent)
    trainval = random.sample(num_list, tv)
    train = random.sample(trainval, tr)

    print("train and val size", tv)
    print("traub suze", tr)
    ftrainval = open(os.path.join(saveBasePath, 'trainval.txt'), 'w')
    ftest = open(os.path.join(saveBasePath, 'test.txt'), 'w')
    ftrain = open(os.path.join(saveBasePath, 'train.txt'), 'w')
    fval = open(os.path.join(saveBasePath, 'val.txt'), 'w')

    for i in num_list:
        name = total_seg[i][:-4] + '\n'
        if i in trainval:
            ftrainval.write(name)
            if i in train:
                ftrain.write(name)
            else:
                fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()
    print("Generate txt in ImageSets done.")

    print("Check datasets format, this may take a while.")
    print("检查数据集格式是否符合要求，这可能需要一段时间。")
    classes_nums = np.zeros([256], np.int32)
    for i in tqdm(num_list):
        name = total_seg[i]
        png_file_name = os.path.join(segfilepath, name)
        if not os.path.exists(png_file_name):
            raise ValueError("未检测到标签图片 %s，请查看具体路径下文件是否存在以及后缀是否为png。" % png_file_name)

        png = np.array(Image.open(png_file_name), np.uint8)
        if len(np.shape(png)) > 2:
            print("标签图片%s的shape为%s，不属于灰度图或者八位彩图，请仔细检查数据集格式。" % (name, str(np.shape(png))))
            print("标签图片需要为灰度图或者八位彩图，标签的每个像素点的值就是这个像素点所属的种类。")

        classes_nums += np.bincount(np.reshape(png, [-1]), minlength=256)

    print("打印像素点的值与数量。")
    print('-' * 37)
    print("| %15s | %15s |" % ("Key", "Value"))
    print('-' * 37)
    for i in range(256):
        if classes_nums[i] > 0:
            print("| %15s | %15s |" % (str(i), str(classes_nums[i])))
            print('-' * 37)

    if classes_nums[255] > 0 and classes_nums[0] > 0 and np.sum(classes_nums[1:255]) == 0:
        print("检测到标签中像素点的值仅包含0与255，数据格式有误。")
        print("二分类问题需要将标签修改为背景的像素点值为0，目标的像素点值为1。")
    elif classes_nums[0] > 0 and np.sum(classes_nums[1:]) == 0:
        print("检测到标签中仅仅包含背景像素点，数据格式有误，请仔细检查数据集格式。")

    print("JPEGImages中的图片应当为.jpg文件、SegmentationClass中的图片应当为.png文件。")

