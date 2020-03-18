"""
    WideResNet model definition
    ported from https://github.com/meliketoy/wide-resnet.pytorch/blob/master/networks/wide_resnet.py
"""

import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import math

__all__ = ['BayesWideResNet28x10']

from models import varinf as vi

def conv3x3(in_planes, out_planes, stride=1, lv_init=-6, var_p=-1):
    return vi.BayesConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True, lv_init=lv_init, var_p=var_p)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=math.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)

class WideBasic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, lv_init=-5, var_p=-1):
        super(WideBasic, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = vi.BayesConv2d(in_planes, planes, kernel_size=3, padding=1, bias=True, lv_init=lv_init, var_p=var_p)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = vi.BayesConv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True, lv_init=lv_init, var_p=var_p)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                vi.BayesConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True, lv_init=lv_init, var_p=var_p),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(F.relu(self.bn1(x))))
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)

        return out


class WideResNet(nn.Module):
    def __init__(self, num_classes=10, depth=28, widen_factor=10, dropout_rate=0., lv_init=-5, var_p=-1):
        super(WideResNet, self).__init__()
        self.in_planes = 16

        assert ((depth - 4) % 6 == 0), 'Wide-resnet depth should be 6n+4'
        n = (depth - 4) / 6
        k = widen_factor

        nstages = [16, 16 * k, 32 * k, 64 * k]

        self.conv1 = conv3x3(3, nstages[0], var_p=var_p)
        self.layer1 = self._wide_layer(WideBasic, nstages[1], n, dropout_rate, stride=1, var_p=var_p)
        self.layer2 = self._wide_layer(WideBasic, nstages[2], n, dropout_rate, stride=2, var_p=var_p)
        self.layer3 = self._wide_layer(WideBasic, nstages[3], n, dropout_rate, stride=2, var_p=var_p)
        self.bn1 = nn.BatchNorm2d(nstages[3], momentum=0.9)
        self.linear = vi.BayesLinear(nstages[3], num_classes, var_p=var_p)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride, var_p=-1):
        strides = [stride] + [1] * int(num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, var_p=var_p))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


class BayesWideResNet28x10:
    base = WideResNet
    args = list()
    kwargs = {'depth': 28, 'widen_factor': 10}
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