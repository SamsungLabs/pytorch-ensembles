import argparse
import numpy as np
import os
import sys
import tabulate
import time
import torch
import torch.nn.functional as F

import models
from utils import snapshot_data as data
from utils import snapshot_transforms as transforms
from utils import fge_utils as utils


parser = argparse.ArgumentParser(description='FGE training')

parser.add_argument('--dir', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--transform', type=str, default='VGG')
parser.add_argument('--data_path', type=str, default='~/datasets')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--num-workers', type=int, default=4)
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--epochs', type=int)
parser.add_argument('--cycle', type=int)
parser.add_argument('--lr_1', type=float)
parser.add_argument('--lr_2', type=float)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--wd', type=float, default=1e-4)
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1 - off)')
parser.add_argument('--iter', type=int)

args = parser.parse_args()

assert (args.cycle % 2 == 0 or args.cycle == 1), 'Cycle length should be even'

os.makedirs(args.dir, exist_ok=True)
with open(os.path.join(args.dir, 'fge.sh'), 'w') as f:
    f.write(' '.join(sys.argv))
    f.write('\n')
    
class Logger():
    def __init__(self):
        self.stdout = sys.stdout  # save it because stdout will be replaced
        
    def write(self, message):
        self.stdout.write(message)
        with open(os.path.join(args.dir, 'stdout_fge_on_pretrained.log'), 'a') as log:
            log.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
sys.stdout = Logger()

def cyclic_learning_rate(epoch, cycle, alpha_1, alpha_2):
    def schedule(iter):
        t = ((epoch % cycle) + iter) / cycle
        if t < 0.5:
            return alpha_1 * (1.0 - 2.0 * t) + alpha_2 * 2.0 * t
        else:
            return alpha_1 * (2.0 * t - 1.0) + alpha_2 * (2.0 - 2.0 * t)
    return schedule

torch.backends.cudnn.benchmark = True

if args.seed != 1:
    print(f'Using given seed {args.seed}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

loaders, num_classes = data.loaders(args.dataset, args.data_path, args.batch_size, args.num_workers,
                                    getattr(transforms, args.transform).train, getattr(transforms, args.transform).test)

architecture = getattr(models, args.model)
model = architecture.base(num_classes=num_classes, **architecture.kwargs)
criterion = F.cross_entropy

checkpoint = torch.load(args.ckpt)
if 'epoch' in checkpoint:
    start_epoch = checkpoint['epoch'] + 1
    model.load_state_dict(checkpoint['model_state'])
else:
    start_epoch = 160
    model.load_state_dict(checkpoint)
model.cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=args.lr_1,
    momentum=args.momentum,
    weight_decay=args.wd
)

ensemble_size = 0
test_preds_sum = np.zeros((len(loaders['test'].dataset), num_classes))

columns = ['ep', 'lr', 'tr_loss', 'tr_acc', 'te_nll', 'te_acc', 'ens_nll', 'ens_acc', 'time']

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll

sample_i = 0
for epoch in range(args.epochs):
    time_ep = time.time()
    lr_schedule = cyclic_learning_rate(epoch, args.cycle, args.lr_1, args.lr_2)
    train_res = utils.train(loaders['train'], model, optimizer, criterion, lr_schedule=lr_schedule)
    test_res = utils.test(loaders['test'], model, criterion)
    time_ep = time.time() - time_ep
    test_preds, test_targets = utils.predictions(loaders['test'], model)  # returns probs
    train_preds, train_targets = utils.predictions(loaders['train'], model)  # returns probs
    # if this epoch is the last before (midcycle point = args.cycle//2)
    if args.cycle == 1 or (epoch % args.cycle + 1) == args.cycle // 2:
        ensemble_size += 1
        test_preds_sum += test_preds
        ens_acc = 100.0 * np.mean(np.argmax(test_preds_sum, axis=1) == test_targets)
        ens_nll = nll(test_preds_sum / ensemble_size, test_targets) / test_preds.shape[0]

        utils.save_checkpoint(
            args.dir,
            sample_i,
            name=f'{args.dataset}-FGE_{args.model}_run_{args.iter}',
            model_state=model.state_dict()
        )
        sample_i += 1

    values = [epoch, lr_schedule(1.0), train_res['loss'], train_res['accuracy'], test_res['nll'],
              test_res['accuracy'], ens_nll, ens_acc, time_ep]
    table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
    if epoch % 40 == 0:
        table = table.split('\n')
        table = '\n'.join([table[1]] + table)
    else:
        table = table.split('\n')[2]
    print(table)
