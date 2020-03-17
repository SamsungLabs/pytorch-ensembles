"""
    implementation of KFAC Laplace, see reference
    base class ported from: https://github.com/Thrandis/EKFAC-pytorch/kfac.py
"""

import torch
import torch.nn.functional as F
import copy
import itertools
import tqdm
import numpy as np


class KFACLaplace(torch.optim.Optimizer):
    r"""KFAC Laplace: based on Scalable Laplace
    Code is partially copied from https://github.com/Thrandis/EKFAC-pytorch/kfac.py.
    """
    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 constraint_norm=False, data_size = 50000, use_batch_norm = False, epochs=1):
        """ K-FAC Preconditionner for Linear and Conv2d layers.
        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.
        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
            use_batch_norm: whether or not batch norm layers should be computed
        """
        if sua:
            raise NotImplementedError
        self.net = net
        self.state = net.state_dict()
        self.mean_state = copy.deepcopy(self.state)
        self.data_size = data_size
        self.use_batch_norm = use_batch_norm
        self.epochs = epochs

        self.eps = eps
        self.pi = pi
        self.update_freq = update_freq
        self.constraint_norm = constraint_norm
        self.params = []
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)

        super(KFACLaplace, self).__init__(self.params, {})

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)

    def cuda(self):
        self.net.cuda()

    def load_state_dict(self, checkpoint, **kwargs):
        self.net.load_state_dict(checkpoint, **kwargs)

        self.mean_state = self.net.state_dict()

    def eval(self):
        self.net.eval()

    def train(self):
        self.net.train()

    def apply(self, *args, **kwargs):
        self.net.apply(*args, **kwargs)

    def sample(self, scale=1.0, **kwargs):
        self.net.load_state_dict(self.mean_state)

        for group in self.params:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            if 'BatchNorm' in group['layer_type'] and self.use_batch_norm:

                z = torch.zeros_like(weight).normal_()
                sample = state['w_ic'].matmul(z)

                if bias is not None:

                    z = torch.zeros_like(bias).normal_()
                    bias_sample = state['b_ic'].matmul(z)

            else:
                # now compute inverse covariances
                ixxt, iggt, ixxt_chol, iggt_chol = self._inv_covs(state['xxt'], state['ggt'], num_locations=state['num_locations'])
                state['ixxt'] = ixxt
                state['iggt'] = iggt

                # draw samples from AZB
                # appendix B of ritter et al.
                s = weight.shape
                if group['layer_type'] == 'Conv2d':
                    s = (s[0], s[1] * s[2] * s[3])
                if bias is not None:
                    s = (s[0], s[1] + 1)
                z = torch.randn(s[0], s[1], device=ixxt.device, dtype=ixxt.dtype)
                sample = iggt_chol.matmul(z.matmul(ixxt_chol.t()))
                sample *= scale # No need to rescale by data_size
                if group['layer_type'] == 'Conv2d':
                    sample *= state['num_locations'] ** 0.5

                if bias is not None:
                    bias_sample = sample[:, -1].contiguous().view(*bias.shape)
                    sample = sample[:, :-1]
            weight.data.add_(sample.view_as(weight))
            if bias is not None:
                bias.data.add_(bias_sample.view_as(bias))

    def step(self, update_stats=True, update_params=True, comp_inv=False):
        #Performs one step of preconditioning.
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]

            if group['layer_type'] in ['Linear', 'Conv2d']:
                # Update convariances and inverses
                if update_stats:
                    if self._iteration_counter % self.update_freq == 0:
                        self._compute_covs(group, state)
                        if comp_inv:
                            ixxt, iggt, _, _ = self._inv_covs(state['xxt'], state['ggt'],
                                                        state['num_locations'])
                            state['ixxt'] = ixxt
                            state['iggt'] = iggt
                    else:
                        self._compute_covs(group, state)
                # Cleaning
                if 'x' in self.state[group['mod']]:
                    del self.state[group['mod']]['x']
                if 'gy' in self.state[group['mod']]:
                    del self.state[group['mod']]['gy']
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                         stride=mod.stride)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t())
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=1.,
                                alpha=1.)
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t())
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=1.,
                                alpha=1.)

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0] / self.epochs
            tg = torch.trace(ggt) * xxt.shape[0] / self.epochs
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps * num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)

        # # Compute cholesky
        xxt_chol = (xxt / self.epochs + torch.diag(diag_xxt)).cholesky()
        ggt_chol = (ggt / self.epochs + torch.diag(diag_ggt)).cholesky()

        # invert cholesky
        xxt_ichol = torch.inverse(xxt_chol)
        ggt_ichol = torch.inverse(ggt_chol)

        # invert matrix
        ixxt = xxt_ichol.t().matmul(xxt_ichol)
        iggt = ggt_ichol.t().matmul(ggt_ichol)

        return ixxt, iggt, xxt_ichol, ggt_ichol

    def laplace_epoch(self, loader, cuda=True, regression=False, verbose=False, subset=None):
        """Go through the dataset to build the Laplace approximation"""
        print('Starting %d epoch(s) to build the Laplace approximation' % self.epochs)
        num_batches = len(loader)
    
        # Need to turn to eval to disable batch normalization and/or dropout
        self.eval()
    
        if subset is not None:
            num_batches = int(num_batches * subset)
            loader = itertools.islice(loader, num_batches)
    
        for ep in range(self.epochs):
            loader = tqdm.tqdm(loader, total=num_batches)
            for i, (input, target) in enumerate(loader):
                if cuda:
                    input = input.cuda(non_blocking=True)
                    target = target.cuda(non_blocking=True)

                output = self.net(input)
                loss = F.cross_entropy(output, target)

                self.zero_grad()
                loss.backward()
                with torch.no_grad():
                    self.step(update_params=False, comp_inv=((i == num_batches - 1) and (ep == self.epochs - 1)))
            print('Epochs done:', ep + 1, '/', self.epochs)

        print('Laplace approximation ready')

