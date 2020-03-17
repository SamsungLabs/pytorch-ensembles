"""
    PreResNet model definition
    ported from https://github.com/bearpaw/pytorch-classification/blob/master/models/cifar/preresnet.py
"""

import torch.nn as nn
import torchvision.transforms as transforms
import math

from models import varinf as vi

__all__ = ['BayesPreResNet110', 'BayesPreResNet164']


def conv3x3(in_planes, out_planes, stride=1, lv_init=-5, var_p=-1):
    return vi.BayesConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False, lv_init=lv_init, var_p=var_p)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, lv_init=-5, var_p=-1):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride, lv_init=lv_init, var_p=var_p)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, lv_init=lv_init, var_p=var_p)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, lv_init=-5, var_p=-1):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = vi.BayesConv2d(inplanes, planes, kernel_size=1, bias=False, lv_init=lv_init, var_p=var_p)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = vi.BayesConv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False, lv_init=lv_init, var_p=var_p)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = vi.BayesConv2d(planes, planes * 4, kernel_size=1, bias=False, lv_init=lv_init, var_p=var_p)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class PreResNet(nn.Module):
    def __init__(self, num_classes=10, depth=110, lv_init=-5, var_p=-1):
        super(PreResNet, self).__init__()
        if depth >= 44:
            assert (depth - 2) % 9 == 0, 'depth should be 9n+2'
            n = (depth - 2) // 9
            block = Bottleneck
        else:
            assert (depth - 2) % 6 == 0, 'depth should be 6n+2'
            n = (depth - 2) // 6
            block = BasicBlock
        self.var_p = var_p


        self.inplanes = 16
        self.conv1 = vi.BayesConv2d(3, 16, kernel_size=3, padding=1,
                               bias=False, var_p=var_p)
        self.layer1 = self._make_layer(block, 16, n, lv_init=lv_init, var_p=var_p)
        self.layer2 = self._make_layer(block, 32, n, stride=2, lv_init=lv_init, var_p=var_p)
        self.layer3 = self._make_layer(block, 64, n, stride=2, lv_init=lv_init, var_p=var_p)
        self.bn = nn.BatchNorm2d(64 * block.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = vi.BayesLinear(64 * block.expansion, num_classes, var_p=var_p)

        for m in self.modules():
            if isinstance(m, vi.BayesConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, lv_init=-5, var_p=-1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                vi.BayesConv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False, lv_init=lv_init, var_p=var_p),
            )

        layers = list()
        layers.append(block(self.inplanes, planes, stride, downsample, lv_init=lv_init, var_p=var_p))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, lv_init=lv_init, var_p=var_p))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)

        x = self.layer1(x)  # 32x32
        x = self.layer2(x)  # 16x16
        x = self.layer3(x)  # 8x8
        x = self.bn(x)
        x = self.relu(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class BayesPreResNet110:
    base = PreResNet
    args = list()
    kwargs = {'depth': 110}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

class BayesPreResNet164:
    base = PreResNet
    args = list()
    kwargs = {'depth': 164}
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
