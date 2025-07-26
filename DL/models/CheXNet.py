# encoding: utf-8

"""
The main CheXNet model implementation.
"""
import torch.nn as nn
import torchvision
import torch


CHEXNET_CKPT_PATH = '/teams/Thymoma_1685081756/PFT/code/WaveletAttention-main/models/CheXNetCKPT/CheXNet.pth.tar'


class DenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size=64):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)

        original_conv = self.densenet121.features.conv0
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        self.densenet121.features.conv0 = new_conv

        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            #nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x


class LightDenseNet121(nn.Module):
    """Model modified.

    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.

    """
    def __init__(self, out_size=64):
        super(LightDenseNet121, self).__init__()
        full_densenet121 = DenseNet121(out_size=out_size)
        checkpoint = torch.load(CHEXNET_CKPT_PATH)
        full_densenet121.load_state_dict(checkpoint['state_dict'], strict=False)

        # self.densenet121 = torchvision.models.densenet121(pretrained=True)

        original_conv = full_densenet121.densenet121.features.conv0
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight = nn.Parameter(original_conv.weight.mean(dim=1, keepdim=True))
        full_densenet121.densenet121.features.conv0 = new_conv
        self.features = nn.Sequential(*list(full_densenet121.densenet121.features.children())[:8])
        self.final_norm = nn.BatchNorm2d(256)
        self.output_linear = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, out_size)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.final_norm(x)
        x = self.output_linear(x)
        return x
