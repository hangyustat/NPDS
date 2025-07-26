from __future__ import absolute_import
from Data import load_process_images_coords_progress, load_process_images_coords_progress_result_domain, NPDSDataset
from torch.utils.data import random_split, DataLoader
from utils.util import seed_everything, worker_init_fn
from functools import partial
from utils.util import AverageMeter, ProgressMeter, accuracy, parse_gpus, meanSquaredError, seed_everything
from models import NPDNet, NPDLoss
from models import DenseNet121, LightDenseNet121
# system lib
import os
import time
import sys
import argparse
# numerical libs
import random

import numpy as np
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchinfo import summary
from utils.checkpoint import save_checkpoint, load_checkpoint
import matplotlib.pyplot as plt
import pickle
CHEXNET_CKPT_PATH = '/teams/Thymoma_1685081756/PFT/code/WaveletAttention-main/models/CheXNetCKPT/CheXNet.pth.tar'


def adjust_learning_rate(optimizer, epoch, base_lr, warmup=False):
    """Adjust the learning rate"""
    if epoch <= 20:
        # lr = 0.00001 if warmup and epoch == 0 else args.base_lr
        lr = 0.00001 if warmup else base_lr
    elif epoch <= 60:
        lr = base_lr * 0.1
    elif epoch <= 80:
        lr = base_lr * 0.01
    elif epoch <= 100:
        lr = base_lr * 0.001
    else:
        lr = base_lr * 0.0001

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

def compute_classification_metrics(y_pred, y_true, threshold=0.5):
    y_pred_label = (y_pred >= threshold).float()

    correct = (y_pred_label == y_true).float().sum()
    accuracy = correct / y_true.numel()

    TP = ((y_pred_label == 1) & (y_true == 1)).sum().float()
    FN = ((y_pred_label == 0) & (y_true == 1)).sum().float()
    sensitivity = TP / (TP + FN + 1e-8)

    return accuracy, sensitivity, TP + FN + 1e-8


def train(net, optimizer, epoch, data_loader, args):
    learning_rate = optimizer.param_groups[0]["lr"]
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('NPDLoss', ':.4f')
    acc_meter = AverageMeter('Accuracy', ':4.6f')
    sens_meter = AverageMeter('Sensitivity', ':4.6f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, acc_meter, sens_meter],
        prefix="Epoch (Train LR {:6.4f}): [{}] ".format(learning_rate, epoch))  # Initialize ProgressMeter

    net.train()  # Set the model to training mode

    tic = time.time()
    if epoch == 1:
        train_pids = []
        train_anno_ids = []
        train_before_dates = []
        train_after_dates = []
    for batch_idx, (images0, images1, images0_n, images1_n, progs, pids, anno_ids, before_dates, after_dates) in enumerate(data_loader):

        images0, images1, images0_n, images1_n, progs = images0.to(args.device, non_blocking=True), \
                                                        images1.to(args.device, non_blocking=True), \
                                                        images0_n.to(args.device, non_blocking=True), \
                                                        images1_n.to(args.device, non_blocking=True), \
                                                        progs.to(args.device, non_blocking=True)
        if epoch == 1:
            train_pids = train_pids + list(pids)
            train_anno_ids = train_anno_ids + list(anno_ids)
            train_before_dates = train_before_dates + list(before_dates)
            train_after_dates = train_after_dates + list(after_dates)

        data_time.update(time.time() - tic)

        optimizer.zero_grad()  # Clear gradients for next training step

        df1, df2, prob = net(images0, images1, images0_n, images1_n)  # Forward pass

        npdLoss = NPDLoss()
        loss = npdLoss(prob, progs, df1, df2, alpha=args.alpha)
        loss.backward()  # Backpropagation to compute gradients
        optimizer.step()  # Update model parameters
        losses.update(loss.item(), df1.size(0))
        acc, sens, pos_n = compute_classification_metrics(prob, progs, threshold=0.5)
        acc_meter.update(acc.item(), df1.size(0))
        sens_meter.update(sens.item(), pos_n.item())

        batch_time.update(time.time() - tic)
        tic = time.time()
        if (batch_idx + 1) % args.disp_iter == 0 or (batch_idx + 1) == len(data_loader):
            epoch_msg = progress.get_message(batch_idx + 1)
            print(epoch_msg)
            args.log_file.write(epoch_msg + "\n")
    if epoch == 1:
        with open(os.path.join(args.ckpt, 'train_meta.pkl'), 'wb') as f:
            pickle.dump({
                'pids': train_pids,
                'anno_ids': train_anno_ids,
                'before_dates': train_before_dates,
                'after_dates': train_after_dates
            }, f)
    return acc_meter.avg, sens_meter.avg, losses.avg


def validate(net, epoch, data_loader, args):
    learning_rate = 0.0
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('NPDLoss', ':.4f')
    acc_meter = AverageMeter('Accuracy', ':4.6f')
    sens_meter = AverageMeter('Sensitivity', ':4.6f')
    progress = ProgressMeter(
        len(data_loader),
        [batch_time, data_time, losses, acc_meter, sens_meter],
        prefix="Epoch (Valid LR {:6.4f}): [{}] ".format(learning_rate, epoch))  # Initialize ProgressMeter

    net.eval()  # Set the model to training mode
    with torch.no_grad():
        tic = time.time()
        if epoch == 1:
            val_pids = []
            val_anno_ids = []
            val_before_dates = []
            val_after_dates = []
        for batch_idx, (images0, images1, images0_n, images1_n, progs, pids, anno_ids, before_dates, after_dates) in enumerate(data_loader):
            if epoch == 1:
                val_pids = val_pids + list(pids)
                val_anno_ids = val_anno_ids + list(anno_ids)
                val_before_dates = val_before_dates + list(before_dates)
                val_after_dates = val_after_dates + list(after_dates)
            images0, images1, images0_n, images1_n, progs = images0.to(args.device, non_blocking=True), \
                                                            images1.to(args.device, non_blocking=True), \
                                                            images0_n.to(args.device, non_blocking=True), \
                                                            images1_n.to(args.device, non_blocking=True), \
                                                            progs.to(args.device, non_blocking=True)

            data_time.update(time.time() - tic)

            df1, df2, prob = net(images0, images1, images0_n, images1_n)  # Forward pass

            npdLoss = NPDLoss()
            loss = npdLoss(prob, progs, df1, df2, alpha=args.alpha)
            losses.update(loss.item(), df1.size(0))
            acc, sens, pos_n = compute_classification_metrics(prob, progs, threshold=0.5)
            acc_meter.update(acc.item(), df1.size(0))
            sens_meter.update(sens.item(), pos_n.item())

            batch_time.update(time.time() - tic)
            tic = time.time()
            if (batch_idx + 1) % args.disp_iter == 0 or (batch_idx + 1) == len(data_loader):
                epoch_msg = progress.get_message(batch_idx + 1)
                print(epoch_msg)
                args.log_file.write(epoch_msg + "\n")
    if epoch == 1:
        with open(os.path.join(args.ckpt, 'val_meta.pkl'), 'wb') as f:
            pickle.dump({
                'pids': val_pids,
                'anno_ids': val_anno_ids,
                'before_dates': val_before_dates,
                'after_dates': val_after_dates
            }, f)
    return acc_meter.avg, sens_meter.avg, losses.avg


def main(args, train_loader, val_loader):
    if len(args.gpu_ids) > 0:
        assert (torch.cuda.is_available())
        cudnn.benchmark = True
        kwargs = {"num_workers": args.workers, "pin_memory": True}
        args.device = torch.device("cuda:{}".format(args.gpu_ids[0]))
    else:
        kwargs = {}
        args.device = torch.device("cpu")

    feature_extractor_CT = DenseNet121(out_size=args.image_feature_dim)
    feature_extractor_N = LightDenseNet121(out_size=args.image_feature_dim)

    checkpoint = torch.load(CHEXNET_CKPT_PATH)
    feature_extractor_CT.load_state_dict(checkpoint['state_dict'], strict=False)

    net = NPDNet(feature_extractor_CT, feature_extractor_N, num_slices=args.CT_slice_num, num_heads=args.num_heads, dp_rate=args.dropout, fdim=args.image_feature_dim)

    optimizer = optim.Adam(net.parameters(), lr=args.base_lr, betas=(args.beta1, 0.999), weight_decay=args.weight_decay)

    start_epoch = 0
    best_loss = 1e8
    best_acc = 0
    best_sens = 0

    for key, val in vars(args).items():
        args.log_file.write("{:16} {}".format(key, val) + "\n")
    args.log_file.write("--------------------------------------------------" + "\n")

    # multi-GPUs
    if len(args.gpu_ids) > 1:
        net = torch.nn.DataParallel(net, args.gpu_ids)

    net.to(args.device)
    summary(net)
    args.log_file.write("Training starts.\n")

    train_accs = []
    val_accs = []

    train_sens = []
    val_sens = []

    train_losses = []
    val_losses = []

    for epoch in range(start_epoch, args.num_epoch):
        adjust_learning_rate(optimizer, epoch, args.base_lr, args.warmup)
        print('lr now: %1.8f' % optimizer.param_groups[0]['lr'])
        args.log_file.write('lr now: %1.8f' % optimizer.param_groups[0]['lr'])

        train_epoch_acc, train_epoch_sens, train_epoch_loss = train(net, optimizer, epoch, train_loader, args)
        val_epoch_acc, val_epoch_sens, val_epoch_loss = validate(net, epoch, val_loader, args)

        train_accs.append(train_epoch_acc)
        val_accs.append(val_epoch_acc)

        train_sens.append(train_epoch_sens)
        val_sens.append(val_epoch_sens)

        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        np.save(args.ckpt + '/train_losses.npy', np.array(train_losses))
        np.save(args.ckpt + '/val_losses.npy', np.array(val_losses))

        np.save(args.ckpt + '/train_accs.npy', np.array(train_accs))
        np.save(args.ckpt + '/val_accs.npy', np.array(val_accs))

        np.save(args.ckpt + '/train_sens.npy', np.array(train_sens))
        np.save(args.ckpt + '/val_sens.npy', np.array(val_sens))

        is_best = val_epoch_loss < best_loss

        best_loss = min(val_epoch_loss, best_loss)
        best_acc = max(val_epoch_acc, best_acc)
        best_sens = max(val_epoch_sens, best_sens)
        print("best loss: ", best_loss)
        print("best acc: ", best_acc)
        print("best sens: ", best_sens)

        save_checkpoint({
            "epoch": epoch + 1,
            "arch": 'DenseNet121',
            "state_dict": net.cpu().state_dict(),  # net.module.cpu().state_dict() will error for no data parallel
            "best_loss": best_loss,
            "best_acc": best_acc,
            "best_sens": best_sens,
            "optimizer": optimizer.state_dict(),
        }, is_best, epoch + 1, save_path=args.ckpt)

        net.to(args.device)

        args.log_file.write("--------------------------------------------------" + "\n")

    args.log_file.write("Training ends.\n")
    print('Job Done.')


if __name__ == '__main__':
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="Copula")

    # Model settings
    parser.add_argument("--num_heads", type=int, default=4,
                        help="The number of heads of Attention Module")
    parser.add_argument("--image_feature_dim", type=int, default=128,
                        help="image feature dimension (default: 128)")
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout rate of MLP")

    # Dataset settings
    parser.add_argument("--workers", default=4, type=int,
                        help="number of data loading works")
    parser.add_argument("--image_folders", default=[], type=list,
                        help="ct image path (default: ./data/lung_mask_20_npy)")
    parser.add_argument("--ct_image_aug", default=False, type=bool,
                        help="when ct_image_aug is true, ct images will be augmented (default: False)")
    parser.add_argument("--image_target_shape", default=(256, 256), type=tuple,
                        help="target shape of CT image (default: (256, 256))")
    parser.add_argument("--CT_slice_num", default=10, type=int,
                        help="slice number of CT image (default: 10)")
    parser.add_argument("--train_val_split", type=float, default=0.2,
                        help="split ratio of train/val dataset (default: 0.2)")

    # Optimizion settings
    parser.add_argument("--gpu_ids", default="0",
                        help="gpus to use, e.g. 0-3 or 0,1,2,3")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="batch size for training and validation (default: 128)")
    parser.add_argument("--num_epoch", type=int, default=200,
                        help="number of epochs to train (default: 200)")
    parser.add_argument("--optim", default="SGD",
                        help="optimizer")
    parser.add_argument("--base_lr", type=float, default=0.01,
                        help="learning rate (default: 0.01)")
    parser.add_argument("--weight_decay", type=float, default=5e-4,
                        help="SGD weight decay (default: 5e-4)")
    parser.add_argument("--warmup", default=False, type=bool,
                        help="warmup for deeper network")
    parser.add_argument("--test_size", default=0.2, type=float,
                        help="test/val dataset split size (default: 0.2)")
    parser.add_argument("--beta1", default=0.9, type=float,
                        help="momentum for sgd, beta1 for adam")
    parser.add_argument("--alpha", default=1.0, type=float,
                        help="alpha for loss")

    # Misc
    parser.add_argument("--seed", type=int, default=1,
                        help="random seed (default: 1)")
    parser.add_argument("--disp_iter", type=int, default=100,
                        help="frequence to display training status (default: 100)")
    parser.add_argument("--ckpts", default="./ckpts/",
                        help="folder to output checkpoints")

    args = parser.parse_args()
    args.gpu_ids = parse_gpus(args.gpu_ids)

    args.image_folders = ['/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealData_60_npy/',
                     '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataAnno20240608/npy_CT/',
                     '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataAnno20240610/npy_CT/',
                     '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataAnno20240623/npy_CT/',
                     '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataUpdate20240802/npy_CT/',
                     '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataUpdate20240805/npy_CT/']


    # args.image_folders = ['/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealData_60_npy/']

    anno_csv_paths = ['/teams/Thymoma_1685081756/NPDS/CT_compare_data/Lung_SegAndCls_20230925_gender_age.csv',
                      '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataAnno20240623/csv/真实标注/真实数据标注-20240623 - 副本.xlsx']

    real_result_csv_paths = ['/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataUpdate20240805/csv/S计算及预测/RealData_S_compute_realdata60(label_fixed-withmmD)_predict_result_20240814.xlsx',
                             '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataUpdate20240805/csv/S计算及预测/RealData_S_compute_0610+0623+0805+0802(triple_check_label_fixed)_predict_result_20240814.xlsx']
    args.seed = 1
    seed_everything(args.seed)

    args.base_lr = 0.0001
    args.alpha = 1.0
    args.batch_size = 4
    args.num_epoch = 200
    args.ckpt = './ckpts/0716/'
    args.ckpt += 'DenseNet121'
    args.ckpt += "-ctsn" + str(args.CT_slice_num)
    args.ckpt += "-canh" + str(args.num_heads)
    args.ckpt += "-fdim" + str(args.image_feature_dim)
    args.ckpt += "-dp" + str(int(args.dropout * 100))
    args.ckpt += "-bslr" + str(int(args.base_lr * 1e5))
    args.ckpt += "-splitsize" + str(args.train_val_split).replace('.', '')
    args.ckpt += "-alpha" + str(args.alpha).replace('.', '')
    args.ckpt += "-seed" + str(args.seed)

    if not os.path.isdir(args.ckpt):
        os.makedirs(args.ckpt)
    args.log_file = open(os.path.join(args.ckpt, "log_file.txt"), mode="w")

    images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates = load_process_images_coords_progress_result_domain(args.image_folders, anno_csv_paths, real_result_csv_paths)
    MyDataSet = NPDSDataset(images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates)

    total_size = len(MyDataSet)
    train_size = int((1.0 - args.train_val_split) * total_size)
    val_size = total_size - train_size
    print(f'Train set size:{train_size}, Val Set size:{val_size}')
    train_dataset, val_dataset = random_split(MyDataSet, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, worker_init_fn=partial(worker_init_fn, seed=args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, worker_init_fn=partial(worker_init_fn, seed=args.seed))

    main(args, train_loader, val_loader)