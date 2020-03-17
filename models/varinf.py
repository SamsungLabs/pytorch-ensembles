import math
import torch

from torch import functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.utils import _pair
from torch.nn.modules.module import Module
from torch._jit_internal import weak_module, weak_script_method
from torch.nn import init
from torch import randn_like
from torch.nn import functional as F


@weak_module
class _BayesConvNd(Module):
    __constants__ = ['stride', 'padding', 'dilation', 'groups', 'bias']

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 padding, dilation, transposed, output_padding, groups, bias, emp_bayes=False, lv_init=-5):

        if transposed:
            raise Exception('_BayesConvNd does not support arg transposed=True.')
        if emp_bayes:
            raise NotImplementedError('emp_bayes is not implemented yet')

        super(_BayesConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.lv_init = lv_init

        self.weight = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))
        self.wlog_sigma = Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size))

        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.wlog_sigma, self.lv_init - 0.1, self.lv_init)

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

@weak_module
class BayesConv2d(_BayesConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, lv_init=-5, var_p=-1):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.var_p=var_p
        super(BayesConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, lv_init=lv_init)

    @weak_script_method
    def forward(self, input):
        eps = torch.randn_like(self.wlog_sigma)
        W = self.weight + eps * torch.exp(self.wlog_sigma)
        return F.conv2d(
            input, W, self.bias, self.stride,
            self.padding, self.dilation, self.groups)

    def kl(self, beta):
        kl = torch.Tensor([0.0]).cuda()
        # kl += torch.sum((self.weight ** 2) / self.var_p) wd is done by optimizer
        kl += beta*torch.sum((torch.exp(self.wlog_sigma)**2) / self.var_p - 1)
        kl -= beta*torch.sum(torch.log((torch.exp(self.wlog_sigma)**2) / self.var_p))

        return 0.5*kl

    def __repr__(self):
        return 'BayesConv2d(lv_init=%s, wshape=%s, var_p=%s)' % (self.lv_init, list(self.weight.shape), self.var_p)


class BayesLinear(Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True, lv_init=-5, var_p=-1):
        super(BayesLinear, self).__init__()
        self.in_features = in_features
        self.lv_init = lv_init
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        self.wlog_sigma = Parameter(torch.Tensor(out_features, in_features))
        self.var_p = var_p
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        init.uniform_(self.wlog_sigma, self.lv_init - 0.1, self.lv_init)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        mu = F.linear(input, self.weight, self.bias)
        sigma = torch.sqrt(F.linear(input ** 2, torch.exp(self.wlog_sigma) ** 2) + 1e-8)

        return mu + sigma * torch.randn_like(mu)

    def kl(self, beta):
        kl = torch.Tensor([0.0]).cuda()
        # kl += torch.sum((self.weight ** 2) / self.var_p) wd is done by optimizer
        kl += beta*torch.sum((torch.exp(self.wlog_sigma)**2) / self.var_p - 1)
        kl -= beta*torch.sum(torch.log((torch.exp(self.wlog_sigma)**2) / self.var_p))

        return 0.5*kl

    def __repr__(self):
        return 'BayesLinear(lv_init=%s, wshape=%s, var_p=%s))' % (self.lv_init, list(self.weight.shape), self.var_p)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None)