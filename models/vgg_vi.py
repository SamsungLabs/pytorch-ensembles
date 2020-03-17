"""
    VGG model definition
    ported from https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import math

import torch.nn as nn
import torchvision.transforms as transforms

from models import varinf as vi

__all__ = ['BayesVGG16BN']


def make_layers(cfg, batch_norm=False, lv_init=-5, var_p=-1):
    layers = list()
    in_channels = 3
    #print('make_layers', lv_init)
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = vi.BayesConv2d(in_channels, v, kernel_size=3, padding=1, lv_init=lv_init, var_p=var_p)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
}


class VGG(nn.Module):
    def __init__(self, num_classes=10, depth=16, batch_norm=False, lv_init=-5, var_p=-1):
        if lv_init > 0:
            raise Exception('init_logvar should be negative')
        super(VGG, self).__init__()
        self.features = make_layers(cfg[depth], batch_norm, lv_init=lv_init, var_p=var_p)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0),
            vi.BayesLinear(512, 512, var_p=var_p),
            nn.ReLU(True),
            nn.Dropout(p=0),
            vi.BayesLinear(512, 512, var_p=var_p),
            nn.ReLU(True),
            vi.BayesLinear(512, num_classes, var_p=var_p),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class Base:
    base = VGG
    args = list()
    kwargs = dict()
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

class BayesVGG16BN(Base):
    kwargs = {'batch_norm': True}


