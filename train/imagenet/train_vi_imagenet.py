import argparse
import os
import time
import torch
import torch.nn.functional as F
import models
import tabulate
import copy
from utils.utils import ens_pred
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from utils import utils
from models import BayesResNet50

parser = argparse.ArgumentParser(description='SGD/SWA training')

parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--fname', type=str, default=None, required=True, help='checkpoint and outputs file name')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--workers', type=int, default=16, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')

parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')

parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.001, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lr_vars', type=float, default=0.001, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--lv_init', type=float, default=-5, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=0, help='weight decay (default: 1e-4)')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--beta', type=float, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--smalr', action='store_true', help='random seed (default: 1)')

global_time = time.time()

args = parser.parse_args()
args.method = ''

print('\nArgs:', args, '\n')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)

torch.backends.cudnn.benchmark = True

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

train_sampler = None
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

train_dataset = datasets.ImageFolder(
    traindir,
    transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ]))

train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.workers, pin_memory=True, sampler=train_sampler)

val_loader = torch.utils.data.DataLoader(
    datasets.ImageFolder(valdir, transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])),
    batch_size=args.batch_size, shuffle=False,
    num_workers=args.workers, pin_memory=True)

loaders = {
    'train': train_loader,
    'test': val_loader
}
num_classes = 1000


var_p = 1./(len(loaders['train'].dataset) * args.wd)
print('prior_variance =', var_p)
print('Preparing model')
model = BayesResNet50(lv_init=args.lv_init, var_p=var_p)
model.cuda()

def schedule(epoch):
    t = (epoch) / (args.epochs/2)
    lr_ratio = 0.01
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    elif t <= 1.0:
        factor = lr_ratio
    else:
        factor = 0.001
    return args.lr_init * factor

criterion = F.cross_entropy

params = [p for n, p in model.named_parameters() if 'wlog_sigma' not in n]
paramsn = [n for n, p in model.named_parameters() if 'wlog_sigma' not in n]
print(paramsn[:10], len(paramsn))
optimizer1 = torch.optim.SGD(
    params,
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)
print(optimizer1)

params = [p for n, p in model.named_parameters() if 'wlog_sigma' in n]
paramsn = [n for n, p in model.named_parameters() if 'wlog_sigma' in n]
print(paramsn[:10], len(paramsn))
optimizer2 = torch.optim.Adam(
    params,
    lr=args.lr_vars,
    weight_decay=0,
)
print(optimizer2)

start_epoch = 0
args.model = args.model.replace('Bayes', '')
fnames = utils.read_models(args, base=os.path.abspath('/home/aashukha/megares'))
fnames = np.random.permutation(fnames)
args.model = 'Bayes' + args.model
# Result form the paper
# fnames = ['~/megares/ImageNet-ResNet50-cn-025--1564562952-1.pth.tar']

print('Resume training from %s' % fnames[0])

checkpoint = torch.load(fnames[0])['state_dict']
state_dict = checkpoint

state_dict_v2 = copy.deepcopy(state_dict)

for key in state_dict:
    state_dict_v2[key[7:]] = state_dict_v2.pop(key)

model.load_state_dict(state_dict_v2, strict=False)
columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'nll', 'kl', 'te_ac', 'te_nll']
test_res = utils.eval(loaders['test'], model, criterion)
print(test_res)

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer1, lr)
    optvar = optimizer2
    train_res = utils.train_epoch_vi(
        loaders['train'], model, criterion,
        optimizer1, args.beta, # if epoch > args.epochs/2 else 0,
        optvar, verbose=True)
    test_res = {'loss': None, 'accuracy': None}

    test_res = utils.eval(loaders['test'], model, criterion)

    if epoch % 5 == 0:
        pth = os.path.join(args.dir, args.fname +'_ep%s.pt' % epoch)
        torch.save(model.state_dict(), pth)
        print('Save to:', pth)

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep, train_res['nll'], train_res['kl'], test_res['accuracy'], test_res['loss']]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)


pth = os.path.join(args.dir, args.fname +'.pt')
torch.save(model.state_dict(), pth)
print('Save to:', pth)
print('Global time: ', time.time() - global_time)

print('Final test res:', test_res)
test_preds, test_targets = utils.predictions(loaders['test'], model)
#np.savez(os.path.join(args.dir, args.fname), test_preds=test_preds, test_targets=test_targets)

model.eval()
print('\nEnsemble results:')
for ns in [1, 10]:
    acc, llh, log_prob, targets, preds = ens_pred(model, loaders['test'], ns)
    print('ns %s: acc %.4f, llh %.4f' % (ns, acc, llh))

print()
print()
print(args.dataset, args.model, args.beta, acc, llh)
print()