import argparse
import os, sys
import time
import tabulate
import torch
import torch.nn.functional as F
import torchvision
import numpy as np

import models
from utils import snapshot_data as data
from utils import snapshot_transforms as transforms
from utils import swag_utils as utils
from utils.swag_model import SWAG

parser = argparse.ArgumentParser(description='SWAG training')
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--data_path', type=str, default='~/datasets')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL',
                    help='model name (default: None)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--save_freq', type=int, default=25, metavar='N', help='save frequency (default: 25)')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N', help='evaluation frequency (default: 5)')
parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
parser.add_argument('--swa_start', type=float, default=161, metavar='N', help='SWA start epoch number (default: 161)')
parser.add_argument('--swa_lr', type=float, default=0.02, metavar='LR', help='SWA LR (default: 0.02)')
parser.add_argument('--swa_c_epochs', type=int, default=1, metavar='N',
                    help='SWA model collection frequency/cycle length in epochs (default: 1)')
parser.add_argument('--cov_mat', action='store_true', help='save sample covariance')
parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')
parser.add_argument('--resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to resume training from (default: None)')
parser.add_argument('--swa_resume', type=str, default=None, metavar='CKPT',
                    help='checkpoint to restor SWA from (default: None)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1 - off)')

args = parser.parse_args()

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'command.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
    
class Logger():
    def __init__(self):
        self.stdout = sys.stdout  # save it because stdout will be replaced
        
    def write(self, message):
        self.stdout.write(message)
        with open(os.path.join(args.dir, 'stdout_run_swag.log'), 'a') as log:
            log.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
sys.stdout = Logger()

torch.backends.cudnn.benchmark = True
if args.seed != 1:
    print(f'Using given seed {args.seed}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(args.dataset, args.data_path, args.batch_size, args.num_workers,
                                    getattr(transforms, args.transform).train, getattr(transforms, args.transform).test)

architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
model = model.cuda()

swag_model = SWAG(architecture.base, no_cov_mat=not args.cov_mat, max_num_models=args.max_num_models, 
                *architecture.args, num_classes=num_classes, **architecture.kwargs)
swag_model = swag_model.cuda()

for (module, name) in swag_model.params:
    cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
    module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt.to(torch.device('cpu')))
    
def criterion(model, input, target):
    output = model(input)
    loss = F.cross_entropy(output, target)
    return loss, output

def schedule(epoch):
    t = epoch / args.swa_start
    # same as with FGE. we do half of the pre-training with constant lr, then linearly decrease lr during the other half.
    lr_ratio = args.swa_lr / args.lr_init
    if t <= 0.5:
        factor = 1.0
    elif t <= 0.9:
        factor = 1.0 - (1.0 - lr_ratio) * (t - 0.5) / 0.4
    else:
        factor = lr_ratio
    # after SWAG kicks in: constant lr
    return args.lr_init * factor

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_init,
    momentum=args.momentum,
    weight_decay=args.wd
)

start_epoch = 0
if args.resume is not None:
    print('Resume training from %s' % args.resume)
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

if args.swa_resume is not None:
    checkpoint = torch.load(args.swa_resume)
    swag_model = SWAG(architecture.base, no_cov_mat=not args.cov_mat, max_num_models=args.max_num_models, 
                      loading=True, *architecture.args, num_classes=num_classes, **architecture.kwargs)
    swag_model = swag_model.cuda()
    swag_model.load_state_dict(checkpoint['state_dict'])

    for (module, name) in swag_model.params:
        cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
        module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt.to(torch.device('cpu')))

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'swa_te_loss', 'swa_te_acc', 'time', 'mem_usage']
swag_res = {'loss': None, 'accuracy': None}

sgd_ens_preds = None
sgd_targets = None
n_ensembled = 0.

for epoch in range(start_epoch, args.epochs):
    time_ep = time.time()

    lr = schedule(epoch)
    utils.adjust_learning_rate(optimizer, lr)
    
    if (epoch + 1) > args.swa_start and args.cov_mat:
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)
    else:
        train_res = utils.train_epoch(loaders['train'], model, criterion, optimizer)

    if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
        test_res = utils.eval(loaders['test'], model, criterion)
    else:
        test_res = {'loss': None, 'accuracy': None}
        
    sgd_res = utils.predict(loaders["test"], model)
    sgd_preds = sgd_res["predictions"]
    sgd_targets = sgd_res["targets"]
#     np.savez(os.path.join(args.dir, f"individual_preds-{epoch+1}.npz"), test_preds=sgd_preds, test_targets=sgd_targets.astype('int64'))

    if (epoch + 1) > args.swa_start and (epoch + 1 - args.swa_start) % args.swa_c_epochs == 0:
        print("updating sgd_ens")
        if sgd_ens_preds is None:
            sgd_ens_preds = sgd_preds.copy()
        else:
            sgd_ens_preds = sgd_ens_preds * n_ensembled / (n_ensembled + 1) + sgd_preds/ (n_ensembled + 1)
        n_ensembled += 1
        swag_model.collect_model(model)
        if epoch == 0 or epoch % args.eval_freq == args.eval_freq - 1 or epoch == args.epochs - 1:
            swag_model.sample(0.0)
            utils.bn_update(loaders['train'], swag_model)
            swag_res = utils.eval(loaders['test'], swag_model, criterion)
        else:
            swag_res = {'loss': None, 'accuracy': None}

    if (epoch + 1) % args.save_freq == 0:
#         utils.save_checkpoint(
#             args.dir,
#             epoch + 1,
#             name='individual',
#             state_dict=model.state_dict(),
#             optimizer=optimizer.state_dict()
#         )
        if (epoch+1) > args.swa_start:
            utils.save_checkpoint(
                args.dir,
                epoch + 1,
                name='swag_ensembled',
                state_dict=swag_model.state_dict(),
            )

    time_ep = time.time() - time_ep
    memory_usage = torch.cuda.memory_allocated()/(1024.0 ** 3)
    values = [epoch + 1, lr, train_res['loss'], train_res['accuracy'], test_res['loss'], test_res['accuracy'],
              swag_res['loss'], swag_res['accuracy'], time_ep, memory_usage]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='8.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)

# utils.save_checkpoint(
#     args.dir,
#     args.epochs,
#     name='individual',
#     state_dict=model.state_dict(),
#     optimizer=optimizer.state_dict()
# )
utils.save_checkpoint(
    args.dir,
    args.epochs,  # cur epoch + 1
    name=f'swag_ensembled',
    state_dict=swag_model.state_dict(),
)
