#!/user/bin/env python3
# -*- coding: utf-8 -*-
"""
@Author: yjf
@Create: 2024/3/23 22:12
@Message: null
"""
import argparse
import os.path
import time

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.unet import Unet
from utils.dataloader import UDataset, unet_dataset_collate
from utils.loss import Focal_Loss, CE_Loss, Dice_loss
from utils.util import generate_dir, set_random_seed, save_info_when_training, CLASSES, set_optimizer_lr
from utils.util_eval import Evaluator
from utils.util_metrics import calculate_score


def run(args, model, device, dataloader, log_dir):
    # 是否给不同种类赋予不同的损失权值，默认是平衡的。设置的话，注意设置成numpy形式的，长度和num_classes一样。
    # 例如 num_classes = 3，则有 cls_weights = np.array([1, 1, 1], np.float32)
    cls_weights = np.ones([args.num_classes], np.float32)
    weights = torch.from_numpy(cls_weights).to(device)

    # 定义评估器
    evaluator = Evaluator(args.input_size, args.num_classes, device, image_txt_path=args.val_txt_path,
                          dataset_path=args.dataset_path, log_dir=log_dir, total_epoch=args.epoch)

    # 根据当前batch_size，自适应调整学习率
    if dataloader['train'].dataset.length > 10000:
        nbs = 16
        lr_limit_max = 1e-4 if args.optimizer_type in ['Adam', 'AdamW'] else 1e-1
        lr_limit_min = 1e-4 if args.optimizer_type in ['Adam', 'AdamW'] else 5e-4
        init_lr_fit = min(max(args.batch_size / nbs * args.init_lr, lr_limit_min), lr_limit_max)
        min_lr_fit = min(max(args.batch_size / nbs * (args.init_lr * args.min_lr_factor), lr_limit_min * 1e-2),
                         lr_limit_max * 1e-2)
    else:
        init_lr_fit = args.init_lr
        min_lr_fit = args.init_lr * args.min_lr_factor

    # 定义优化器
    optimizer = {'Adam': torch.optim.Adam(model.parameters(), init_lr_fit, betas=(0.9, 0.999), weight_decay=args.weight_decay),
                 'AdamW': torch.optim.AdamW(params=model.parameters(), lr=init_lr_fit, weight_decay=args.weight_decay)
                 }[args.optimizer_type]

    # # 学习率调节器
    # exp_lr_scheduler = {
    #     'MultiStepLR': lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma),
    #     'CosineAnnealingWarmRestarts': lr_scheduler.CosineAnnealingWarmRestarts(
    #         optimizer, T_0=args.epoch, T_mult=1, eta_min=min_lr_fit)
    # }[args.scheduler_type]

    # 每个epoch的训练信息
    loss_list, val_loss_list, val_f_score_list, lr_list = [], [], [], []
    best_epoch = 0
    evaluator_mIoU = 0.0

    # 训练模型
    start_time = time.time()
    for epoch in range(args.epoch):
        print('-' * 38)
        print('Epoch {}/{}'.format(epoch, args.epoch - 1))
        current_lr = optimizer.state_dict()['param_groups'][0]['lr']
        lr_list.append(current_lr)
        print('current model optimizer of lr:{}'.format(current_lr))

        # 解冻模型主干权重
        if args.freeze_train and epoch >= args.freeze_epoch:
            model.freeze_backbone(False)
            args.freeze_train = False
            print("已解冻所有模型权重")

        # every epoch has train and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                # lr_scheduler.step()
                # set train model
                model.train()
            else:
                model.eval()

            running_loss = []
            dice_list = []
            mIoU_list = []
            for batch in tqdm(dataloader[phase], desc=phase + ' Data Processing Progress'):
                imgs, pngs, labels = batch
                imgs = imgs.to(device)
                pngs = pngs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                # forward propagation, track history if only train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(imgs)
                    # 计算损失 默认使用 CE Loss + Dice Loss 组合
                    if args.focal_loss:
                        loss = Focal_Loss(outputs, pngs, weights, num_classes=args.num_classes)
                    else:
                        loss = CE_Loss(outputs, pngs, weights, num_classes=args.num_classes)

                    if args.dice_loss:
                        loss_dice = Dice_loss(outputs, labels)
                        loss = loss + loss_dice

                    # backward and optimized only in the training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        dice_coefficient, mIoU, precision, recall = calculate_score(outputs, labels)
                        dice_list.append(dice_coefficient)
                        mIoU_list.append(mIoU)

                # statistics loss
                running_loss.append(loss.item())

            # 当前epoch 的loss dice&mIoU (计算的结果有问题)
            epoch_loss = sum(running_loss) / len(running_loss)
            epoch_dice = sum(dice_list) / len(dice_list) if len(dice_list) > 0 else 0.0
            epoch_mIoU = sum(mIoU_list) / len(mIoU_list) if len(mIoU_list) > 0 else 0.0

            # save data / learning rate regulator
            if phase == 'train':
                loss_list.append(epoch_loss)
                print('{} Loss:{:.8f} '.format(phase, epoch_loss))
            else:
                # 计算当前 epoch 的 mIoU Precision
                evaluator_mIoU = evaluator.on_epoch_end(epoch, model_eval=model, classes_eval=CLASSES, draw_info=True)
                val_loss_list.append(epoch_loss)
                val_f_score_list.append(evaluator_mIoU)
                print('{} Loss:{:.8f} dice:{:.4f} mIou:{:.4f} evaluator_mIoU:{:.4f}'
                      .format(phase, epoch_loss, epoch_dice, epoch_mIoU, evaluator_mIoU))
                # 保存当前最好模型权重
                if args.best_acc <= evaluator_mIoU:
                    args.best_acc = evaluator_mIoU
                    best_epoch = epoch
                    torch.save(model.state_dict(), os.path.join(log_dir, 'best_model_weight.pth'))
                    print("已保存当前最好模型权重：", os.path.join(log_dir, 'best_model_weight.pth'))

                save_info_when_training(loss_list, val_loss_list, lr_list, log_dir)

            with open(os.path.join(log_dir, "epoch_loss.txt"), 'a', encoding='utf8') as f:
                if phase == 'train':
                    f.write('\n' + '-' * 42 + '\n')
                f.write('Epoch {}/{},\tphase:{}\ncurrent_lr:{:.6f},\tloss:{:.6f}\n'
                        .format(epoch, args.epoch - 1, phase, current_lr, epoch_loss))

            # 保存最后的模型权重，并绘制训练曲线
            if phase == 'val' and epoch == args.epoch - 1:
                torch.save(model.state_dict(), os.path.join(log_dir, '{}_loss{:.4f}_dice{:.4f}.pth'
                                                            .format(epoch, epoch_loss, evaluator_mIoU)))

        # 调整当前训练的学习率
        set_optimizer_lr(optimizer, epoch, args.epoch, warmup_epochs=args.freeze_epoch // 2, max_lr=init_lr_fit, min_lr=min_lr_fit)

    # 最后保存模型训练信息
    time_spend = time.time() - start_time
    print('\nThe total time spent training the model is {:.0f}h{:.0f}m{:.0f}s.'.format(
        time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))
    with open(os.path.join(log_dir, "epoch_loss.txt"), 'a', encoding='utf8') as f:
        f.write('\nThe save best weight epoch {}.'.format(best_epoch))
        f.write('\nThe total time spent training the model is {:.0f}h{:.0f}m{:.0f}s.\n'.format(
            time_spend // 3600, time_spend % 3600 // 60, time_spend % 60))
        f.write("\ntrain_loss:")
        f.write(",".join(str(round(_, 4)) for _ in loss_list))
        f.write("\nval_loss:")
        f.write(",".join(str(round(_, 4)) for _ in val_loss_list))


def build_model(args, device):
    """
    创建模型，加载预训练权重并冻结主干
    :param args: 模型配置参数
    :param device: 推理设备
    :return:
    """
    # 实例化模型
    model = Unet(backbone_type=args.backbone_type, num_classes=args.num_classes, head_up=args.head_up)
    model.to(device)
    if args.weight_path is not None:
        print("Model weights loading：{}".format(args.weight_path))
        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(args.weight_path, map_location='cpu')
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # 显示没有匹配上的Key
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("Fail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # 训练冻结模型主干权重
    if args.freeze_train:
        model.freeze_backbone(True)
        print("已冻结主干部分权重")

    return model


def build_dataloader(args):
    """
    创建数据集加载器
    :param args: 数据集加载器配置参数
    :return:
    """
    # 读取数据集对应的txt
    with open(os.path.join(args.dataset_path, args.train_txt_path), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(args.dataset_path, args.val_txt_path), "r") as f:
        val_lines = f.readlines()

    dataset = {'train': UDataset(train_lines, num_classes=args.num_classes, train=True,
                                 dataset_path=args.dataset_path),
               'val': UDataset(val_lines, num_classes=args.num_classes, train=False,
                               dataset_path=args.dataset_path)}
    dataloader = {
        'train': DataLoader(dataset['train'], batch_size=args.batch_size, shuffle=True,
                            num_workers=args.num_workers,
                            collate_fn=unet_dataset_collate, pin_memory=True),
        'val': DataLoader(dataset['val'], batch_size=args.batch_size, shuffle=False,
                          num_workers=args.num_workers,
                          collate_fn=unet_dataset_collate, pin_memory=True)}
    # 训练集/验证集的大小
    dataset_size = {x: len(dataset[x]) for x in ['train', 'val']}
    print("Dataset statistical results：")
    print('training dataset loaded with length : {}'.format(dataset_size['train']))
    print('validation dataset loaded with length : {}'.format(dataset_size['val']))

    return dataloader, dataset_size


def save_train_info(args, dataset_sizes):
    """
    保存模型训练参数信息
    :param args: 训练参数配置
    :param dataset_sizes: 数据集大小
    :return:
    """
    current_time = time.strftime("%Y%m%d-%H_%M", time.localtime())
    log_dir = os.path.join(args.log_path, current_time)
    generate_dir(log_dir)
    # 读取配置文件
    argsDict = args.__dict__
    with open(os.path.join(log_dir, 'train_config.txt'), 'w', encoding='utf8') as f:
        f.write("---------training configuration information---------\n")
        f.write("current time : {}.\n".format(current_time))
        f.writelines('dataset train size : {}, val size : {}.\n\n'.format(dataset_sizes['train'], dataset_sizes['val']))
        f.writelines('------------------ start ------------------' + '\n')
        for eachArg, value in argsDict.items():
            f.writelines(eachArg + ' : ' + str(value) + '\n')
        f.writelines('------------------- ok -------------------')
    return log_dir


if __name__ == '__main__':
    print("train")
    parser = argparse.ArgumentParser(description='Unet Train Info')
    parser.add_argument('--info', default=['dev'],
                        help='模型修改备注信息')
    parser.add_argument('--log_path', default='../experiment/train_prod',
                        help='训练信息保存路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='random seed')
    parser.add_argument('--device', default='cuda:1',
                        help='device')
    parser.add_argument('--input_size', default=(512, 512),
                        help='the model input image size')
    parser.add_argument('--best_acc', default=0.2,
                        help='best_acc.')
    # ======================= 网络结构参数=============================
    parser.add_argument('--backbone_type', type=str, default="resnet50",
                        help='选择主干网络')
    parser.add_argument('--num_classes', type=int, default=4,
                        help='目标类别数，对应网络的输出特征通道数')
    parser.add_argument('--weight_path', default=None,
                        help='pre-training model load path.')
    parser.add_argument('--head_up', type=str, default="JUp_Concat",
                        help='选择头网络的多尺度特征融合方式，可选: [unetUp, JUp, JUp_Add, JUp_Concat]')
    # ======================= 加载数据相关参数=============================
    parser.add_argument('--dataset_path', type=str, default="/home/amax/Jiao/dataset/diabetic_first_dev",
                        help='数据集路径')
    parser.add_argument('--train_txt_path', type=str, default="ImageSets/Segmentation/train.txt",
                        help='train 划分的 txt 文件路径')
    parser.add_argument('--val_txt_path', type=str, default="ImageSets/Segmentation/val.txt",
                        help='val 划分的 txt 文件路径')
    # ======================= 训练参数=============================
    parser.add_argument('--epoch', type=int, default=100,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', default=8,
                        help='batch size when training.')
    parser.add_argument('--num_workers', default=8,
                        help='load data num_workers when training.')
    parser.add_argument('--freeze_train', type=bool, default=True,
                        help='freeze backbone when start train.')
    parser.add_argument('--freeze_epoch', type=int, default=10,
                        help='模型冻结主干训练的epoch')
    parser.add_argument('--init_lr', default=0.0001, type=float,
                        help='初始学习率，0.0001 是训练的默认值')
    parser.add_argument('--min_lr_factor', default=0.01, type=float,
                        help='当使用 CosineAnnealingWarmRestarts 时的最小学习率')
    # ======================= 损失函数相关参数=============================
    parser.add_argument('--focal_loss', default=False, type=bool,
                        help='是否使用 Focal Loss 以防止正负样本不平衡，若不选择则默认使用 CE Loss')
    parser.add_argument('--dice_loss', default=True, type=bool,
                        help='是否使用 Dice Loss 根据label额外计算损失，使用后效果更好')
    # ======================= 优化器相关参数=============================
    parser.add_argument('--optimizer_type', default='AdamW', type=str,
                        help='选择使用的优化器，可选(Adam, AdamW, SGD)')
    parser.add_argument('--weight_decay', default=1e-2, type=float,
                        help='Adam 优化器的参数 weight decay')
    # 学习率调节器相关参数
    parser.add_argument('--scheduler_type', default='Cosine_Warmup', type=str,
                        help='选择使用的学习率调节器，可选(MultiStepLR, CosineAnnealingWarmRestarts)')
    parser.add_argument('--lr_steps', default=[10, 18], type=list,
                        help='当使用 MultiStepLR 时，在指定步长进行学习率调节')
    parser.add_argument('--lr_gamma', default=0.1, type=float,
                        help='当使用 MultiStepLR 时，学习率调节系数，即当前学习率乘以该系数')
    train_args = parser.parse_args()

    # 设置随机种子，保证结果的可复现
    set_random_seed(train_args.seed)
    # 选择训练模型使用的设备
    train_device = torch.device(train_args.device if torch.cuda.is_available() else "cpu")
    print("Inference device: ", torch.cuda.get_device_name(train_device))

    # 实例化模型，并加载预训练权重和冻结主干
    train_model = build_model(args=train_args, device=train_device)

    # 加载数据集
    train_dataloader, train_sizes = build_dataloader(args=train_args)

    # 保存运行相关的参数信息
    train_log_dir = save_train_info(args=train_args, dataset_sizes=train_sizes)

    # 开始训练
    run(
        args=train_args,
        model=train_model,
        device=train_device,
        dataloader=train_dataloader,
        log_dir=train_log_dir,
    )
