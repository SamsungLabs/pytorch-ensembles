import os
import time
import glob
import models
import tqdm
import numpy as np
import pandas as pd
import argparse
import torchvision

import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as tv_models

from metrics import metrics_kfold
from scipy.special import logsumexp


def get_sd(fname, args):
    sd = torch.load(fname)
    if 'model_state' in sd:
        return sd['model_state']
    if 'state_dict' in sd:
        return sd['state_dict']
    return sd

def get_targets(loader, args):
    targets = []
    os.makedirs('./.megacache', exist_ok=True)
    targets_f = './.megacache/%s-targets.npy' % args.dataset
    if not os.path.exists(targets_f):
        for _, target in loader:
            targets += [target]
        targets = np.concatenate(targets)
        print('Save targets to %s' % targets_f)
        np.save(targets_f, targets)
    else:
        print('\033[93m' + 'Warning: load default targets from %s' % targets_f + '\033[0m')
        targets = np.load(targets_f)

    return targets

class Logger:
    def __init__(self, base='./logs/'):
        self.res = []
        self.base = base
        os.makedirs(base, exist_ok=True)
        self.df = None

    def add(self, ns, metrics, args, info='', end='\n', silent=False):
        for m in metrics:
            self.res += [[args.dataset, args.model, args.method, ns, m, metrics[m], info]]
        if not silent:
            print('ns %s: acc %.4f, nll %.4f' % (ns, metrics['acc'], metrics['ll']), flush=True, end=end)

    def save(self, args, silent=True):
        self.df = pd.DataFrame(
            self.res, columns=['dataset', 'model', 'method', 'n_samples', 'metric', 'value', 'info'])
        dir = '%s-%s-%s-%s.cvs' % (args.dataset, args.model, args.method, args.fname)
        dir = os.path.join(self.base, dir)
        if not silent:
            print('Saved to:', dir, flush=True)
        self.df.to_csv(dir)

    def print(self):
        print(self.df, flush=True)

    def add_metrics_ts(self, ns, log_probs, targets, args, time_=0):

        if args.dataset == 'ImageNet':
            disable = ('misclass_MI_auroc', 'sce', 'ace')
            n_runs = 2
        else:
            n_runs = 5
            disable = ('misclass_MI_auroc', 'sce', 'ace', 'misclass_entropy_auroc@5', 'misclass_confidence_auroc@5')
        log_prob = logsumexp(np.dstack(log_probs), axis=2) - np.log(ns+1)
        metrics = metrics_kfold(log_prob, targets, n_splits=2, n_runs=n_runs, disable=disable)
        silent = (ns != 0 and (ns + 1) % 10 != 0)
        self.add(ns+1, metrics, args, silent=silent, end=' ')

        args.method = args.method + ' (ts)'
        metrics_ts = metrics_kfold(log_prob, targets, n_splits=2, n_runs=n_runs, temp_scale=True, disable=disable)
        self.add(ns+1, metrics_ts, args, silent=True)
        args.method = args.method[:-5]
        if not silent:
            print("time: %.3f" % (time.time() - time_))

def get_model(args):
    if args.dataset == 'ImageNet':
        if 'vi' in args.method:
            return models.BayesResNet50(lv_init=2, var_p=2).cuda()
        else:
            model = tv_models.__dict__['resnet50']()
            model = torch.nn.DataParallel(model).cuda()
    else:
        model_cfg = getattr(models, args.model)
        model = model_cfg.base(*model_cfg.args, num_classes=args.num_classes, **model_cfg.kwargs).cuda()

    return model

def read_models(args, base='/home/aashukha/megares', run=-1):
    method2file = {
        'onenet': '', 'deepens': '', 'dropout': '', 'vi': '',
        'csgld': 'cSGLD_', 'sse': 'SSE_', 'fge': 'FGE_', 'swag': 'SWAG_'}
    method = method2file[args.method.replace('_augment', '')]

    pattern = '%s-%s%s%s-*.pt*' % (args.dataset, method, args.model, ('_run_%s' % run) if run != -1 else '')
    pattern = os.path.join(base, pattern)
    fnames = glob.glob(pattern)

    if not fnames:
        raise Exception('No models were found with pattern "%s" so check it out.' % pattern)
    print('Readed %s models with pattern %s \n' % (len(fnames), pattern), flush=True)

    return fnames

def get_parser_ens():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--dataset', metavar='DATASET', type=str, required=True,
        help='dataset name CIFAR10/CIFAR100/ImageNet')
    parser.add_argument(
        '--data_path', metavar='PATH', type=str, default='~/data',
        help='path to a data-folder')
    parser.add_argument(
        '--models_dir', metavar='PATH', type=str, default='~/megares',
        help='a dir that stores pre-trained models')
    parser.add_argument(
        '--aug_test', action='store_true', default=False,
        help='enables test-time augmentation')
    parser.add_argument(
        '--batch_size', metavar='N', type=int, default=256,
        help='input batch size')
    parser.add_argument(
        '--num_workers', metavar='M', type=int, default=10,
        help='number of workers')
    parser.add_argument(
        '--fname', metavar='FNAME', type=str, default='unnamed', required=False,
        help='comment to a log file name')

    return parser

def get_parser():
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument(
        '--dir', type=str, default='~', help='training directory (default: None)')
    parser.add_argument(
        '--log_dir', type=str, default='./logs/')
    parser.add_argument(
        '--models_dir', type=str, default='~/megares')
    parser.add_argument(
        '--fname', type=str, default=None, required=True, help='checkpoint and outputs file name')
    parser.add_argument(
        '--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
    parser.add_argument(
        '--data_path', type=str, default='../data', metavar='PATH',
        help='path to datasets location (default: None)')
    parser.add_argument(
        '--batch_size', type=int, default=1024, metavar='N', help='input batch size (default: 128)')
    parser.add_argument(
        '--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
    parser.add_argument(
        '--model', type=str, default=None, required=False, metavar='MODEL',
        help='model name (default: None)')
    parser.add_argument(
        '--resume', type=str, default=None, metavar='CKPT',
        help='checkpoint to resume training from (default: None)')
    parser.add_argument(
        '--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
    parser.add_argument(
        '--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
    parser.add_argument(
        '--lr_init', type=float, default=0.1, metavar='LR',
        help='initial learning rate (default: 0.01)')
    parser.add_argument(
        '--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument(
        '--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--aug_test', action='store_true', default=False)

    return parser

def get_data(args, verbose=False):
    # Only validation data!
    print('Loading dataset %s from %s' % (args.dataset, args.data_path))

    if args.dataset in ['CIFAR10', 'CIFAR100']:
        method_name = args.method.split('_')[0]
        if method_name in ['fge', 'swag', 'sse', 'csgld']:
            from utils.snapshot_transforms import get_transform
            transforms_ = get_transform(method_name, args.model)
            transform_train_cifar = transforms_.train
            transform_test_cifar = transforms_.test
        else:
            model_cfg = getattr(models, args.model)
            transform_train_cifar = model_cfg.transform_train
            transform_test_cifar = model_cfg.transform_test

        if verbose:
            print('Using the following transforms:')
            print('transform_train', transform_train_cifar)
            print('transform_test', transform_test_cifar)

        ds = getattr(torchvision.datasets, args.dataset)
        path = os.path.join(args.data_path, args.dataset.lower())
        transform_test = transform_train_cifar if args.aug_test else transform_test_cifar
        print('Test-time augmentation is ' +
              (('\033[92m' + 'ON' + '\033[0m') if args.aug_test else ('\033[91m' + 'OFF' + '\033[0m')))

        try:
            train_set = ds(path, train=True, download=False, transform=transform_train_cifar)
            test_set = ds(path, train=False, download=False, transform=transform_test)
        except:
            train_set = ds(path, train=True, download=True, transform=transform_train_cifar)
            test_set = ds(path, train=False, download=True, transform=transform_test)

        loaders = {
            'train': torch.utils.data.DataLoader(
                train_set,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                pin_memory=True
            ),
            'test': torch.utils.data.DataLoader(
                test_set,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                pin_memory=True
            )
        }
        num_classes = np.max(train_set.targets)+1
    elif args.dataset == 'ImageNet':
        path = os.path.join(args.data_path, 'imagenet', 'raw-data', 'val')
        print('Assumes %s is a path to ImageNet!' % path)
        if args.aug_test:
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(path,
                transforms.Compose([
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
        else:
            val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(path, transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])),
                batch_size=args.batch_size, shuffle=False,
                num_workers=args.num_workers, pin_memory=True)
        print('Test-time augmentation is ' +
              (('\033[92m' + 'ON' + '\033[0m') if args.aug_test else ('\033[91m' + 'OFF' + '\033[0m')))
        loaders = {'test': val_loader}
        num_classes = 1000
    else:
        raise Exception('Unknown dataset "%s"' % args.dataset)

    return loaders, num_classes

def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def save_checkpoint(dir, epoch, **kwargs):
    state = {'epoch': epoch}
    state.update(kwargs)
    filepath = os.path.join(dir, 'checkpoint-%d.pt' % epoch)
    torch.save(state, filepath)

def train_epoch(loader, model, criterion, optimizer):
    loss_sum = 0.0
    correct = 0.0

    model.train()

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)
        loss = criterion(output, target_var)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * input.size(0)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
    }

def kldiv(model, beta):
    kl = torch.Tensor([0.0]).cuda()
    for c in model.children():
        if hasattr(c, 'children'):
            kl += kldiv(c, beta)
        if hasattr(c, 'kl'):
            kl += c.kl(beta)
    return kl

def train_epoch_vi(loader, model, criterion, optimizer, beta, opt2, verbose=False):
    loss_sum = 0.0
    correct = 0.0
    nll_sum = 0.0
    kl_sum = 0.0

    model.train()
    k = 0

    for i, (input, target) in enumerate(loader):
        k += 1
        input = input.cuda()
        target = target.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        output = model(input_var)

        nll = criterion(output, target_var)
        kl =  kldiv(model, beta)/len(loader.dataset)
        loss = nll + kl

        optimizer.zero_grad()
        if opt2: opt2.zero_grad()
        loss.backward()
        optimizer.step()
        if opt2: opt2.step()

        loss_sum += loss.item() * input.size(0)
        nll_sum += nll.item() * input.size(0)
        kl_sum += kl.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target_var.data.view_as(pred)).sum().item()

        if verbose and i % 500 == 0:
            print('batch', i,
                  'accuracy %.3f' % (100 * correct / (k*256)),
                  'loss %.3f' % (loss_sum / (k*256)),
                  'kl', (kl_sum / (k*256)))

    return {
        'loss': loss_sum / len(loader.dataset),
        'accuracy': correct / len(loader.dataset) * 100.0,
        'nll': nll_sum / len(loader.dataset),
        'kl': kl_sum / len(loader.dataset)
    }

def remove_bar():
    import sys
    sys.stdout.write("\033[F") #back to previous line
    sys.stdout.write("\033[K") #clear line

def one_sample_pred(loader, model, **kwargs):
    preds = []

    for i, (input, target) in enumerate(loader):
        input = input.cuda()
        output = model(input, **kwargs)
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        preds.append(log_probs.cpu().data.numpy())

    return np.vstack(preds)


def ens_pred(model, loader, n_samples):
    targets, log_probs = [], []
    for _ in tqdm(range(n_samples)):
        with torch.no_grad():
            test_preds = one_sample_pred(loader, model)
        log_probs.append(test_preds)
    remove_bar()
    log_probs = np.dstack(log_probs)
    log_prob = logsumexp(log_probs, axis=2) - np.log(n_samples)

    for _, target in loader:
        targets += [target]
    targets = np.concatenate(targets)

    llh = np.mean(log_prob[np.arange(log_prob.shape[0]), targets])
    acc = np.mean(np.argmax(log_prob, axis=1) == targets)
    preds = np.argmax(log_prob, axis=1)
    return acc, llh, log_prob, targets, preds


def eval(loader, model, criterion):
    loss_sum = 0.0
    correct = 0.0

    model.eval()

    with torch.no_grad():
        for i, (input, target) in enumerate(loader):
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            output = model(input_var)
            loss = criterion(output, target_var)

            loss_sum += loss.item() * input.size(0)
            pred = output.data.max(1, keepdim=True)[1]
            correct += (pred == target_var.data.view_as(pred)).sum().item()

        return {
            'loss': loss_sum / len(loader.dataset),
            'accuracy': correct / len(loader.dataset) * 100.0,
        }

def predictions(test_loader, model, **kwargs):
    model.eval()
    preds = []
    targets = []
    for input, target in test_loader:
        input = input.cuda()
        output = model(input, **kwargs)
        log_probs = torch.nn.functional.log_softmax(output, dim=1)
        preds.append(log_probs.cpu().data.numpy())
        targets.append(target.numpy())
    return np.vstack(preds), np.concatenate(targets)
