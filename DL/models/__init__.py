import functools
from .CheXNet import DenseNet121, LightDenseNet121
import torch
import torch.nn as nn
from linformer import Linformer
from .attention_module import attention_module
import torch.nn.functional as F

CHEXNET_CKPT_PATH = '/teams/Thymoma_1685081756/PFT/code/WaveletAttention-main/models/CheXNetCKPT/CheXNet.pth.tar'


class NPDNet(nn.Module):
    def __init__(self, feature_extractor_CT, feature_extractor_N, output_dim_y=1, num_slices=10, num_heads=4, dp_rate=0.2, fdim=128, ddim=128):
        super(NPDNet, self).__init__()
        self.num_slices = num_slices
        self.dp_rate = dp_rate
        self.output_dim_y = output_dim_y
        self.feature_extractor_CT = feature_extractor_CT
        self.feature_extractor_N = feature_extractor_N
        self.num_heads = num_heads
        self.fdim = fdim

        self.relu = nn.ReLU(inplace=False)
        self.gelu = nn.GELU()
        self.ln_img0 = nn.LayerNorm(self.fdim)
        self.ln_img1 = nn.LayerNorm(self.fdim)
        self.ln_imgn0 = nn.LayerNorm(self.fdim)
        self.ln_imgn1 = nn.LayerNorm(self.fdim)
        self.diff_CA = attention_module(fdim, ddim, num_heads=self.num_heads)
        self.conv1d = nn.Conv1d(in_channels=self.num_slices, out_channels=1, kernel_size=1)
        self.classify_layer = nn.Sequential(
            nn.Linear(ddim, ddim // 2),
            nn.ReLU(),
            nn.Dropout(p=self.dp_rate),
            nn.Linear(ddim // 2, ddim // 4),
            nn.ReLU(),
            nn.Dropout(p=(self.dp_rate / 2.0)),
            nn.Linear(ddim // 4, output_dim_y),
            nn.Sigmoid()
        )

    def forward(self, img0, img1, imgn0, imgn1):
        batch_size = img0.shape[0]
        imgs0 = img0.view(-1, 1, img0.shape[2], img0.shape[3])
        imgs1 = img1.view(-1, 1, img1.shape[2], img1.shape[3])
        imgsn0 = imgn0.view(-1, 1, imgn0.shape[2], imgn0.shape[3])
        imgsn1 = imgn1.view(-1, 1, imgn1.shape[2], imgn1.shape[3])

        imgs0_features = self.feature_extractor_CT(imgs0).view(batch_size, self.num_slices, self.fdim)
        imgs0_features = self.ln_img0(imgs0_features)
        imgs1_features = self.feature_extractor_CT(imgs1).view(batch_size, self.num_slices, self.fdim)
        imgs1_features = self.ln_img1(imgs1_features)

        imgsn0_features = self.feature_extractor_N(imgsn0).view(batch_size, self.num_slices, self.fdim)
        imgsn0_features = self.ln_img0(imgsn0_features)
        imgsn1_features = self.feature_extractor_N(imgsn1).view(batch_size, self.num_slices, self.fdim)
        imgsn1_features = self.ln_img1(imgsn1_features)

        diff0_features, _ = self.diff_CA(imgs0_features, imgsn0_features, imgsn0_features)
        diff1_features, _ = self.diff_CA(imgs1_features, imgsn1_features, imgsn1_features)
        diff0_features = self.conv1d(diff0_features).squeeze(1)
        diff1_features = self.conv1d(diff1_features).squeeze(1)

        dd_features = diff1_features - diff0_features

        progress_prob = self.classify_layer(dd_features)
        progress_prob = progress_prob.squeeze(1)

        return diff0_features, diff1_features, progress_prob


class NPDLoss(nn.Module):
    def __init__(self):
        super(NPDLoss, self).__init__()

    def forward(self, pred, label, feat1, feat2, alpha=1.0):

        bce = F.binary_cross_entropy(pred, label)

        cos_sim = F.cosine_similarity(feat1, feat2, dim=1)
        cos_dist = 1.0 - cos_sim

        contrastive = label * cos_dist + (1 - label) * (1.0 - cos_dist)
        contrastive = contrastive.mean()

        return bce + alpha * contrastive


