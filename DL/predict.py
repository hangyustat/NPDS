from __future__ import absolute_import
from Data import load_process_images_coords_progress_result_domain, NPDSDataset
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

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score
import seaborn as sns
import pandas as pd

CHEXNET_CKPT_PATH = '/teams/Thymoma_1685081756/PFT/code/WaveletAttention-main/models/CheXNetCKPT/CheXNet.pth.tar'


def find_best_threshold(y_true, y_prob, thresholds, mode='bl'):
    best_thresh = None
    best_score = -np.inf if mode in ['yd', 'f2'] else np.inf

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        sensitivity = tp / (tp + fn + 1e-8)
        specificity = tn / (tn + fp + 1e-8)

        if mode == 'yd':
            score = sensitivity + specificity - 1
            if score > best_score:
                best_score = score
                best_thresh = thresh

        elif mode == 'bl':
            score = abs(sensitivity - specificity)
            if score < best_score:
                best_score = score
                best_thresh = thresh

        elif mode == 'f2':
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            if precision + recall == 0:
                f2 = 0
            else:
                f2 = (5 * precision * recall) / (4 * precision + recall)
            if f2 > best_score:
                best_score = f2
                best_thresh = thresh

        else:
            raise ValueError("mode must be one of ['yd', 'bl', 'f2']")

    return best_thresh


def bootstrap_best_threshold(y_true, y_prob, thresholds_roc, n_bootstrap=20, mode='bl'):
    # thresholds = np.linspace(0, 1, 1000)
    thresholds = thresholds_roc
    best_thresholds = []
    for _ in range(n_bootstrap):
        indices = np.random.choice(len(y_true), size=len(y_true), replace=True)
        y_true_sample = y_true[indices]
        y_prob_sample = y_prob[indices]
        best_thresh = find_best_threshold(y_true_sample, y_prob_sample, thresholds, mode=mode)
        best_thresholds.append(best_thresh)
    return np.median(best_thresholds), best_thresholds


def plot_roc_curves(predicted_prob, true_label, save_path='', bestt=False, title_name='', n_bootstrap=50):

    plt.figure(figsize=(5, 5))

    fpr, tpr, thresholds_roc = roc_curve(true_label, predicted_prob)
    # print("len(thresholds_roc): ", len(thresholds_roc))
    # print(thresholds_roc)
    roc_auc = auc(fpr, tpr)
    np.save(save_path + f'{title_name}_fprs.npy', np.array(fpr))
    best_thresh_yd, best_thresh_f2, best_thresh_bl = 0, 0, 0
    if bestt:
        best_thresh_yd, _ = bootstrap_best_threshold(true_label, predicted_prob, thresholds_roc, mode='yd', n_bootstrap=n_bootstrap)
        best_thresh_f2, _ = bootstrap_best_threshold(true_label, predicted_prob, thresholds_roc, mode='f2', n_bootstrap=n_bootstrap)
        best_thresh_bl, _ = bootstrap_best_threshold(true_label, predicted_prob, thresholds_roc, mode='bl', n_bootstrap=n_bootstrap)

    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')

    # if bestt:
    #     plt.scatter(fpr[idx], tpr[idx], marker='o', color='red', s=80,
    #                 label=f'Optimal {best_thre_mode} Threshold (J={J[idx]:.2f})\nThreshold={optimal_threshold:.6f}')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{title_name} ROC Curve')
    plt.legend(loc="lower right", frameon=True)

    plt.tight_layout()
    if not os.path.exists(save_path + 'ROC_curves/'):
        os.mkdir(save_path + 'ROC_curves/')
    plt.savefig(save_path + 'ROC_curves/' + f'{title_name}_ROC_Curves.png', dpi=600)
    # plt.show()
    plt.close()
    if bestt:
        print(f"Task {i + 1} Optimal Threshold (youden Index): {best_thresh_yd:.8f}")
        print(f"Task {i + 1} Optimal Threshold (f2 Index): {best_thresh_f2:.8f}")
        print(f"Task {i + 1} Optimal Threshold (balance Index): {best_thresh_bl:.8f}")
        np.save(save_path + f'{title_name}_yd_best_thres.npy', np.array(best_thresh_yd))
        np.save(save_path + f'{title_name}_f2_best_thres.npy', np.array(best_thresh_f2))
        np.save(save_path + f'{title_name}_bl_best_thres.npy', np.array(best_thresh_bl))
    return best_thresh_yd, best_thresh_f2, best_thresh_bl


def calculate_metrics(y_true, y_prob, threshold, save_path='', title_name='', task_id=1, save_fig=False, y_pred=np.array([])):
    if y_pred.size == 0:
        y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Pred 0', 'Pred 1'],
                yticklabels=['True 0', 'True 1'])
    plt.title(f'{title_name} Confusion Matrix Task{task_id}', fontsize=12)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    if not os.path.exists(save_path + 'confusion_matrix/'):
        os.mkdir(save_path + 'confusion_matrix/')
    if save_fig:
        plt.savefig(save_path + 'confusion_matrix/' + f'{title_name}_confusion_matrix_Task{task_id}.png', dpi=600)
    # plt.show()
    plt.close()
    return accuracy, npv, ppv, specificity, sensitivity


def calculate_all_metrics(predicted_prob, true_label, best_thresholds, save_path='', title_name='', save_fig=False):
    acc, npv, ppv, spec, sens = calculate_metrics(true_label, predicted_prob, best_thresholds, save_path=save_path, title_name=title_name, task_id=1, save_fig=save_fig)
    results={
        'Task': f'Task_{1}',
        'Accuracy': acc,
        'NPV': npv,
        'PPV': ppv,
        'Specificity': spec,
        'Sensitivity': sens,
        'best_threshold': best_thresholds
    }
    return results


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
    real_result_csv_paths = [
        '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataUpdate20240805/csv/S计算及预测/RealData_S_compute_realdata60(label_fixed-withmmD)_predict_result_20240814.xlsx',
        '/teams/Thymoma_1685081756/NPDS/CT_compare_data/RealDataUpdate20240805/csv/S计算及预测/RealData_S_compute_0610+0623+0805+0802(triple_check_label_fixed)_predict_result_20240814.xlsx']

    real_result_df_part1 = pd.read_excel(real_result_csv_paths[0]).copy()
    real_result_df_part2 = pd.read_excel(real_result_csv_paths[1]).copy()
    real_result_df_part1["set"] = ""
    real_result_df_part1["DL预测"] = np.nan
    real_result_df_part1['patient_id'] = real_result_df_part1['patient_id'].astype(str)
    real_result_df_part2["set"] = ""
    real_result_df_part2["DL预测"] = np.nan
    real_result_df_part2['patient_id'] = real_result_df_part2['patient_id'].astype(str).str.zfill(10)
    real_result_df_part2['检查日期'] = real_result_df_part2['检查日期'].astype(str)
    real_result_df_part2['比较检查日期'] = real_result_df_part2['比较检查日期'].astype(str)

    real_result_df_list = [real_result_df_part1, real_result_df_part2]

    args.seed = 1
    seed_everything(args.seed)

    args.base_lr = 0.0001
    args.alpha = 0.1
    args.batch_size = 4
    args.num_epoch = 200

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
    net = NPDNet(feature_extractor_CT, feature_extractor_N, num_slices=args.CT_slice_num, num_heads=args.num_heads,
                 dp_rate=args.dropout, fdim=args.image_feature_dim)

    model_path = './ckpts/0716/DenseNet121-ctsn10-canh4-fdim128-dp20-bslr10-splitsize02-alpha10-seed1/'
    summary(net)
    checkpoint = torch.load(model_path + 'model_best_checkpoint.pth.tar')
    checkpoint_name = 'ckpt_best'
    if not os.path.exists(model_path + 'result/'):
        os.mkdir(model_path + 'result/')
    result_save_path = model_path + 'result/' + checkpoint_name + '/'
    if not os.path.exists(result_save_path):
        os.mkdir(result_save_path)
    net.load_state_dict(checkpoint['state_dict'], strict=True)
    net.to(args.device)

    images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates = load_process_images_coords_progress_result_domain(
        args.image_folders, anno_csv_paths, real_result_csv_paths)
    MyDataSet = NPDSDataset(images0, images1, coords, progs, pids, anno_ids, before_dates, after_dates)

    total_size = len(MyDataSet)
    train_size = int((1.0 - args.train_val_split) * total_size)
    val_size = total_size - train_size
    print(f'Train set size:{train_size}, Val Set size:{val_size}')
    train_dataset, val_dataset = random_split(MyDataSet, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              worker_init_fn=partial(worker_init_fn, seed=args.seed))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            worker_init_fn=partial(worker_init_fn, seed=args.seed))

    data_loader_list = [val_loader, train_loader]
    filename_list = ['val', 'train']
    filename_list = [checkpoint_name + '_' + fn for fn in filename_list]
    best_thresholds = []
    n_bootstrap = 200
    for i, data_loader in enumerate(data_loader_list):
        filename = filename_list[i]
        predicted_prob = []
        true_label = []
        NPDS_label = []
        with torch.no_grad():
            for batch_idx, (images0, images1, images0_n, images1_n, progs, pids, anno_ids, before_dates, after_dates) in enumerate(data_loader):
                images0, images1, images0_n, images1_n, progs = images0.to(args.device, non_blocking=True), \
                                                                images1.to(args.device, non_blocking=True), \
                                                                images0_n.to(args.device, non_blocking=True), \
                                                                images1_n.to(args.device, non_blocking=True), \
                                                                progs.to(args.device, non_blocking=True)
                _, _, prob = net(images0, images1, images0_n, images1_n)  # Forward pass
                for j in range(len(pids)):
                    prob_now = float(prob[j])
                    true_label_now = int(progs[j])
                    pid = str(pids[j])
                    anno_id = int(anno_ids[j])
                    before_date = str(before_dates[j])
                    after_date = str(after_dates[j])
                    if before_date == '':
                        filter_condition = (real_result_df_part1["nodule_id"] == anno_id) & (real_result_df_part1["patient_id"] == pid)
                        real_result_df_part1.loc[filter_condition, "set"] = filename.split('_')[-1]
                        real_result_df_part1.loc[filter_condition, "DL预测"] = prob_now
                        NPDS_pre = real_result_df_part1.loc[filter_condition, "预测结果"].values[0]
                    else:
                        filter_condition = (real_result_df_part2["结节编号"] == anno_id) & (
                                    real_result_df_part2["patient_id"] == pid) & (real_result_df_part2["检查日期"] == after_date) & (real_result_df_part2["比较检查日期"] == before_date)
                        real_result_df_part2.loc[filter_condition, "set"] = filename.split('_')[-1]
                        real_result_df_part2.loc[filter_condition, "DL预测"] = prob_now
                        NPDS_pre = real_result_df_part2.loc[filter_condition, "预测结果"].values[0]
                    NPDS_label.append(NPDS_pre)
                    predicted_prob.append(prob_now)
                    true_label.append(true_label_now)
        # predicted_prob = np.concatenate(predicted_prob).astype(np.float32)
        # true_label = np.concatenate(true_label).astype(np.int32)
        true_label = np.array(true_label, dtype=np.int32)
        predicted_prob = np.array(predicted_prob, dtype=np.float32)
        NPDS_label = np.array(NPDS_label, dtype=np.int32)
        print(f'{filename} set size: ', true_label.size)
        if i == 0:
            bestt = True
            best_thresholds_yd, best_thresholds_f2, best_thresholds_bl = plot_roc_curves(predicted_prob, true_label,
                                                                                         save_path=result_save_path,
                                                                                         bestt=bestt,
                                                                                         title_name=filename_list[i],
                                                                                         n_bootstrap=n_bootstrap)
        else:
            bestt = False
            plot_roc_curves(predicted_prob, true_label, save_path=result_save_path, bestt=bestt,
                            title_name=filename_list[i])

        metrics_yd = calculate_all_metrics(predicted_prob, true_label, best_thresholds_yd,
                                           title_name=filename_list[i] + '_yd', save_path=result_save_path,
                                           save_fig=True)

        metrics_f2 = calculate_all_metrics(predicted_prob, true_label, best_thresholds_f2,
                                           title_name=filename_list[i] + '_f2', save_path=result_save_path,
                                           save_fig=True)

        metrics_bl = calculate_all_metrics(predicted_prob, true_label, best_thresholds_bl,
                                           title_name=filename_list[i] + '_bl', save_path=result_save_path,
                                           save_fig=True)

        metrics_dict_list = [metrics_yd, metrics_f2, metrics_bl]

        npds_accuracy, npds_npv, npds_ppv, npds_specificity, npds_sensitivity = calculate_metrics(true_label, 0, 0, save_fig=False, y_pred=NPDS_label)
        print(f'NPDS acc:{npds_accuracy}')
        print(f'NPDS npv:{npds_npv}')
        print(f'NPDS ppv:{npds_ppv}')
        print(f'NPDS spe:{npds_specificity}')
        print(f'NPDS sen:{npds_sensitivity}')

        npds_metrics_dict = {
            "Accuracy": [npds_accuracy],
            "NPV": [npds_npv],
            "PPV": [npds_ppv],
            "Specificity": [npds_specificity],
            "Sensitivity": [npds_sensitivity]
        }

        npds_df_metrics = pd.DataFrame(npds_metrics_dict)

        if not os.path.exists(result_save_path + 'metrics/'):
            os.mkdir(result_save_path + 'metrics/')

        npds_df_metrics.to_excel(result_save_path + 'metrics/' + f'npds_metrics_{filename}.xlsx', index=False)

        index_name_list = ['yd', 'f2', 'bl']
        print("----------------------------------------------------------------------")
        for k, m in enumerate(metrics_dict_list):
            print("\n")
            print("Optimal Index", index_name_list[k])
            print("\n")
            print(filename_list[i] + ' set: ')
            print(
                f"Task {m['Task']}: Accuracy={m['Accuracy']:.3f}, NPV={m['NPV']:.3f}, PPV={m['PPV']:.3f}, Specificity={m['Specificity']:.3f}, Sensitivity={m['Sensitivity']:.3f}")

            mdf = pd.DataFrame([m])

            mdf.to_excel(
                result_save_path + 'metrics/' + f'{index_name_list[k]}_{filename_list[i]}_class_metrics_results.xlsx',
                index=False)

        print("----------------------------------------------------------------------")
        print("Job Done.")

    real_result_df_part1.to_excel(
        result_save_path + 'real_result_part1_DL.xlsx',
        index=False)
    real_result_df_part2.to_excel(
        result_save_path + 'real_result_part2_DL.xlsx',
        index=False)
    print("Job Done.")
