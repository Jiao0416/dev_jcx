#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/4/22 11:16
@Message: null
"""
import argparse
import copy
import os

import cv2
import numpy as np
import torch
from PIL import Image
import torch.nn.functional as F

from models.unet import Unet
from utils.util import cvtColor, preprocess_input, resize_image, add_info_to_image
from utils.util import CLASSES as name_classes, COLORS_CV as colors


if __name__ == "__main__":
    print("predict")
    parser = argparse.ArgumentParser(description='Unet Predict Info')
    parser.add_argument('--image_dir', default='data/diabetic_orig/H00002.jpg', help='需要预测的文件路径')
    parser.add_argument('--device', default='cuda:0', help='推理设备')
    parser.add_argument('--input_size', default=(512, 512), help='the model input image size')
    parser.add_argument('--backbone_type', type=str, default="resnet50", help='选择主干网络')
    parser.add_argument('--num_classes', type=int, default=4, help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--head_up', type=str, default="unetUp", help='选择头网络的多尺度特征融合方式')
    parser.add_argument('--weight_path',
                        default="/home/amax/Jiao/experiment/train_dev/20240422-10_06/best_model_weight.pth", help='加载训练好的模型权重')
    parser.add_argument('--save', default="output", help='预测结果保存路径')
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 实例化UNet网络
    unet = Unet(backbone_type=args.backbone_type, num_classes=args.num_classes, head_up=args.head_up)
    unet.to(device)
    if args.weight_path is not None:
        unet.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=True)
        print("Model weights loaded：{}".format(args.weight_path))
    unet.eval()

    basename = os.path.basename(args.image_path)
    image = Image.open(args.image_path)
    # 在这里将图像转换成RGB图像，防止灰度图在预测时报错。
    image = cvtColor(image)
    # 对输入图像进行一个备份，后面用于绘图
    old_img = copy.deepcopy(image)
    original_h = np.array(image).shape[0]
    original_w = np.array(image).shape[1]
    # 给图像增加灰条，实现不失真的resize
    image_data, nw, nh = resize_image(image, args.input_size)
    # 进行维度扩充，即添加上batch_size维度
    image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

    with torch.no_grad():
        # 数据格式转换 np -> tensor -> tensor gpu/cpu
        images = torch.from_numpy(image_data).to(device)
        # 模型对输入的图片进行推理
        outputs = unet(images)
        pr = outputs[0]

        # 取出每一个像素点的种类
        pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
        #   将灰条部分截取掉
        pr = pr[int((args.input_size[0] - nh) // 2): int((args.input_size[0] - nh) // 2 + nh),
                int((args.input_size[1] - nw) // 2): int((args.input_size[1] - nw) // 2 + nw)]
        # 进行图片的resize
        pr = cv2.resize(pr, (original_w, original_h), interpolation=cv2.INTER_LINEAR)
        # 取出每一个像素点的种类
        pr = pr.argmax(axis=-1)