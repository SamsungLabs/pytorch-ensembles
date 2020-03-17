import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import time
import tqdm
from metrics import metrics_kfold
import sys
from kfacl.utils import Logger

from kfacl import data, utils
import models
from kfacl.laplace import KFACLaplace

import torchvision.transforms as transforms
import torchvision.models as tv_models
import copy

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=True,
                    help='Path to the checkpoint')
parser.add_argument('--dataset', type=str, default=None, required=True,
                    help='Dataset name (tested for CIFAR10, CIFAR100 and ImageNet)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--use_validation', dest='use_validation', action='store_true',
                    help='use validation set instead of test (default: False)')
parser.add_argument('--batch_size', type=int, default=None, metavar='N',
                    help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=None, metavar='N',
                    help='number of workers (default: 4 for CIFARs, 16 for ImageNet)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (tested for VGG16BN, PreResNet110, PreResNet164, WideResNet28x10 for CIFAR,'
                         'ResNet50 for ImageNet)')
parser.add_argument('--fname', type=str, default='', required=False,
                    help='Log file suffix')
parser.add_argument('--logits_dir', type=str, default=None, required=False,
                    help='directory to save final ensemble logits')
parser.add_argument('--N', type=int, default=100,
                    help='Number of samples to ensemble')
parser.add_argument('--scale', type=float, default=1.0,
                    help='Scale factor for the approximate posterior noise')
parser.add_argument('--wd', type=float, default=None,
                    help='L2 penalty with appropriate scaling'
                         '(typically the conventional weight decay, scaled by the size of the dataset)')
parser.add_argument('--gs_low', type=float, default=-2,
                    help='Lower bound for noise scale gridsearch (log10 scale)')
parser.add_argument('--gs_high', type=float, default=1,
                    help='Upper bound for noise scale gridsearch (log10 scale)')
parser.add_argument('--gs_num', type=int, default=25,
                    help='Number of points for gridsearch')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--scale_search', action='store_true',
                    help='Search for optimal noise scale')
parser.add_argument('--test_da', action='store_true',
                    help='Enable test-time data augmentation')

args = parser.parse_args()
args.method = 'KFACLaplace'

if args.wd is None:
    if args.dataset == 'ImageNet' and args.model == 'ResNet50':
        args.wd = 128.1167
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        if args.model in ['PreResNet110', 'PreResNet164']:
            args.wd = 15
        elif args.model in ['VGG16BN', 'WideResNet28x10']:
            args.wd = 25
        else:
            print('Set --wd manually')
            raise NotImplementedError
    else:
        print('Set --wd manually')
        raise NotImplementedError

if args.num_workers is None:
    if args.dataset == 'ImageNet' and args.model == 'ResNet50':
        args.num_workers = 16
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        args.num_workers = 4
    else:
        print('Set --num_workers manually')
        raise NotImplementedError

if args.batch_size is None:
    if args.dataset == 'ImageNet' and args.model == 'ResNet50':
        args.batch_size = 128
    elif args.dataset in ['CIFAR10', 'CIFAR100']:
        args.batch_size = 200
    else:
        print('Set --batch_size manually')
        raise NotImplementedError

print(args.scale)

eps = 1e-12

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if args.dataset == 'ImageNet':
    transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    loaders = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        (transform_train if args.test_da else transform_test),
        (transform_train if args.test_da else transform_test),
    )
    num_classes = 1000
else:
    model_cfg = getattr(models, args.model)

    loaders, num_classes = data.loaders(
        args.dataset,
        args.data_path,
        args.batch_size,
        args.num_workers,
        (model_cfg.transform_train if args.test_da else model_cfg.transform_test),
        (model_cfg.transform_train if args.test_da else model_cfg.transform_test),
        use_validation=args.use_validation,
        shuffle_train=False
    )

if args.dataset == 'ImageNet':
    model = tv_models.__dict__['resnet50']()
else:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
model.cuda()

checkpoint = torch.load(args.file)
if args.dataset == "ImageNet":
    checkpoint = checkpoint['state_dict']
    state_dict = checkpoint
    state_dict_v2 = copy.deepcopy(state_dict)
    for key in state_dict:
        state_dict_v2[key[7:]] = state_dict_v2.pop(key)
    model.load_state_dict(state_dict_v2, strict=False)
else:
    model.load_state_dict(checkpoint)

print("Using KFACLaplace")
print('Train size:', len(loaders['train'].dataset))
model = KFACLaplace(model, eps=args.wd, data_size=len(loaders['train'].dataset), sua=False, pi=True, epochs=1)
model.net.load_state_dict(model.mean_state)
model.laplace_epoch(loaders['train'])

predictions = np.zeros((len(loaders['test'].dataset), num_classes, args.N))
targets = np.zeros(len(loaders['test'].dataset))

if args.scale_search:
    model.net.load_state_dict(model.mean_state)
    model.laplace_epoch(loaders['train'])

    with torch.no_grad():
        optimal_scale = utils.scale_grid_search(loaders['test'], model, num_classes, num_ens=args.N, scale_range=10**np.linspace(args.gs_low, args.gs_high, args.gs_num))
    sys.exit(0)

logger = Logger(base='./logs/')

if args.test_da:
    args.method += '_augment'
for i in tqdm.tqdm(range(args.N)):
    start_time = time.time()

    model.sample(scale=args.scale)

    model.eval()

    k = 0
    with torch.no_grad():
        for input, target in loaders['test']:
            input = input.cuda(non_blocking=True)
            torch.manual_seed(i)
            output = model(input)
            predictions[k:k+input.size()[0], :, i] += F.softmax(output, dim=1).cpu().numpy()
            targets[k:(k+target.size(0))] = target.numpy()
            k += input.size()[0]

    preds = np.mean(predictions[:, :, 0:(i+1)], axis=2)
    trgts = targets.astype(int)
    ll_ens = np.log(1e-12 + preds[np.arange(len(targets)), targets.astype(int)]).mean()
    metrics = metrics_kfold(np.log(preds), trgts, temp_scale=False)
    logger.add(i+1, metrics, args, silent=True)

    # Temperature scaling
    args.method = args.method + ' (ts)'
    metrics_ts = metrics_kfold(np.log(preds), trgts, temp_scale=True)
    logger.add(i+1, metrics_ts, args, silent=True)
    args.method = args.method[:-5]

    logger.save(args)

    if i == args.N - 1 and args.logits_dir is not None:
        os.makedirs(args.logits_dir, exist_ok=True)
        logits_fname = 'logits_%s-%s-%s-%s-%s' % (args.dataset, args.model, args.method, i+1, hash(args.file))
        logits_pth = os.path.join(args.logits_dir, logits_fname)
        log_prob = np.log(1e-12 + preds)
        np.save(logits_pth, log_prob)
        print('Saved to ' + logits_pth)
logger.print()
