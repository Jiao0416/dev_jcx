#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/26 19:53
@Message: null
"""
import os
import cv2
import shutil
import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm
from matplotlib import pyplot as plt

from utils.util import cvtColor, resize_image, preprocess_input, generate_save_epoch
from utils.util_metrics import compute_mIoU


class Evaluator(object):
    def __init__(self, input_shape, num_classes, device, image_txt_path, dataset_path, log_dir, total_epoch: int,
                 eval_flag=True,
                 miou_out_path=".temp_miou_out"):
        super(Evaluator, self).__init__()

        self.net = None
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.dataset_path = dataset_path
        self.log_dir = log_dir
        self.device = device
        self.miou_out_path = miou_out_path
        self.eval_flag = eval_flag
        self.total_epoch = total_epoch
        self.epoch_list = generate_save_epoch(total_epoch)

        with open(os.path.join(dataset_path, image_txt_path), "r") as f:
            image_lines = f.readlines()

        self.image_ids = [image_id.split()[0] for image_id in image_lines]
        self.mious = [0]
        self.epochs = [0]
        self.mPrecision = [0]
        self.mRecall = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_mIoU.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")

    def get_miou_png(self, image):
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image = cvtColor(image)
        orininal_h = np.array(image).shape[0]
        orininal_w = np.array(image).shape[1]
        # ---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        # ---------------------------------------------------------#
        image_data, nw, nh = resize_image(image, (self.input_shape[1], self.input_shape[0]))
        # ---------------------------------------------------------#
        #   添加上batch_size维度
        # ---------------------------------------------------------#
        image_data = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, np.float32)), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            images = images.to(self.device)

            # ---------------------------------------------------#
            #   图片传入网络进行预测
            # ---------------------------------------------------#
            pr = self.net(images)[0]
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = F.softmax(pr.permute(1, 2, 0), dim=-1).cpu().numpy()
            # --------------------------------------#
            #   将灰条部分截取掉
            # --------------------------------------#
            pr = pr[int((self.input_shape[0] - nh) // 2): int((self.input_shape[0] - nh) // 2 + nh), \
                 int((self.input_shape[1] - nw) // 2): int((self.input_shape[1] - nw) // 2 + nw)]
            # ---------------------------------------------------#
            #   进行图片的resize
            # ---------------------------------------------------#
            pr = cv2.resize(pr, (orininal_w, orininal_h), interpolation=cv2.INTER_LINEAR)
            # ---------------------------------------------------#
            #   取出每一个像素点的种类
            # ---------------------------------------------------#
            pr = pr.argmax(axis=-1)

        image = Image.fromarray(np.uint8(pr))
        return image

    def on_epoch_end(self, epoch, model_eval, classes_eval=None, draw_info=True):
        mIoU = 0.0
        if ((epoch in self.epoch_list) or epoch == self.total_epoch - 1) and self.eval_flag:
            self.net = model_eval
            gt_dir = os.path.join(self.dataset_path, "SegmentationClass/")
            pred_dir = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
            for image_id in tqdm(self.image_ids):
                # -------------------------------#
                #   从文件中读取图像
                # -------------------------------#
                image_path = os.path.join(self.dataset_path, "JPEGImages/" + image_id + ".jpg")
                image = Image.open(image_path)
                # ------------------------------#
                #   获得预测txt
                # ------------------------------#
                image = self.get_miou_png(image)
                image.save(os.path.join(pred_dir, image_id + ".png"))

            print("Calculate mIoU.")
            # 执行计算mIoU的函数
            _, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, self.image_ids, self.num_classes,
                                                         classes_eval, save_info=self.log_dir)
            temp_miou = np.nanmean(IoUs) * 100
            temp_mPrecision = round(np.nanmean(Precision) * 100, 2)
            temp_mPA = round(np.nanmean(PA_Recall) * 100, 2)
            mIoU = temp_miou

            self.mious.append(temp_miou)
            self.mPrecision.append(temp_mPrecision)
            self.mRecall.append(temp_mPA)
            self.epochs.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_mIoU.txt"), 'a') as f:
                f.write("epoch: {}, \tmIoU: {}, \tmPA_Recall:{}%, \tmPrecision:{}%\n\n"
                        .format(epoch, str(temp_miou), str(temp_mPA), str(temp_mPrecision)))

            if draw_info:
                plt.figure()
                plt.plot(self.epochs, self.mious, 'MediumPurple', marker='^', linestyle='-', label='mIoU', zorder=3)
                plt.plot(self.epochs, self.mPrecision, '#FF6A6A', marker='*', linestyle='-', label='mPrecision', zorder=2)
                plt.plot(self.epochs, self.mRecall, '#C1CDC1', marker='.', linestyle='-', label='mRecall', zorder=1)

                plt.grid(True)
                plt.xlabel('Epochs')
                plt.ylabel('Metrics')
                plt.title('Metrics Curve')
                plt.legend()
                plt.savefig(os.path.join(self.log_dir, "epoch_mIoU.png"))
                plt.cla()
                plt.close("all")

            print("Get mIoU done.")
            shutil.rmtree(self.miou_out_path)
        # 返回当前 epoch的mIoU
        return mIoU
