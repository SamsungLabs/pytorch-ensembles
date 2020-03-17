import argparse
import os
import time
import torch
import torch.nn.functional as F
import torchvision
import models
import tabulate
from utils import utils


parser = argparse.ArgumentParser(description='Training')

parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--fname', type=str, default=None, required=True, help='checkpoint and outputs file name')

parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
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

print('\nArgs:', args, '\n')

print('Preparing directory %s' % args.dir)
os.makedirs(args.dir, exist_ok=True)
args.method = ''
torch.backends.cudnn.benchmark = True

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
ds = getattr(torchvision.datasets, args.dataset)
path = os.path.join(args.data_path, args.dataset.lower())
train_set = ds(path, train=True, download=True, transform=model_cfg.transform_train)
test_set = ds(path, train=False, download=True, transform=model_cfg.transform_test)
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
num_classes = 10 if args.dataset == 'CIFAR10' else 100

var_p = 1./(len(loaders['train'].dataset) * args.wd)

print('Preparing model')
model = model_cfg.base(*model_cfg.args, num_classes=num_classes, lv_init=args.lv_init, var_p=var_p, **model_cfg.kwargs)
model.cuda()
print(model)

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
optimizer1 = torch.optim.SGD(
    params,
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)
print(optimizer1)

params = [p for n, p in model.named_parameters() if 'wlog_sigma' in n]
optimizer2 = torch.optim.Adam(
    params,
    lr=args.lr_vars,
    weight_decay=0,
)
print(optimizer2)

start_epoch = 0
args.model = args.model.replace('Bayes', '')
fnames = utils.read_models(args, base=os.path.abspath('/home/aashukha/megares'))
args.model = 'Bayes' + args.model

print('Resume training from %s' % fnames[0])
checkpoint = torch.load(fnames[0])
model.load_state_dict(checkpoint, strict=False)

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'time', 'nll', 'kl', 'te_ac', 'te_nll']

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer1, lr)
    optvar = optimizer2
    train_res = utils.train_epoch_vi(
        loaders['train'], model, criterion,
        optimizer1, args.beta, # if epoch > args.epochs/2 else 0,
        optvar)
    test_res = {'loss': None, 'accuracy': None}

    test_res = utils.eval(loaders['test'], model, criterion)

    time_ep = time.time() - time_ep
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'], time_ep, train_res['nll'], train_res['kl'], test_res['accuracy'], test_res['loss']]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

print('Final test res:', test_res)
test_preds, test_targets = utils.predictions(loaders['test'], model)
pth = os.path.join(args.dir, args.fname +'.pt')
print('Save to:', pth)
torch.save(model.state_dict(), pth)
print('Global time: ', time.time() - global_time)


model.eval()
print('\nEnsemble results:')
for ns in [1, 10, 100]:
    acc, llh, log_prob, targets, preds = utils.ens_pred(model, loaders['test'], ns)
    print('ns %s: acc %.4f, llh %.4f' % (ns, acc, llh))

print('\n\n', args.dataset, args.model, args.beta, acc, llh, '\n')
