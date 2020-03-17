import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import random

import models
from utils import snapshot_data as data
from utils import snapshot_transforms as transforms

parser = argparse.ArgumentParser(description='cSG-MCMC training')
parser.add_argument('--dir', type=str, default=None, required=True, help='path to save checkpoints (default: None)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--transform', type=str, default='CIFAR100_CSGMCMC')
# let alpha=1 => momentum=0 => SGD
# let alpha=0.05 => Momentum with momentum=0.95
parser.add_argument('--alpha', type=float, default=1,
                    help='1: SGLD; <1: SGHMC') 
parser.add_argument('--device_id',type = int, help = 'device id to use')
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--cycle_epochs',type = int)
parser.add_argument('--cycles',type = int)
parser.add_argument('--max_lr',type = float)
parser.add_argument('--cycle_saves',type = int)
parser.add_argument('--noise_epochs',type = int, default=None)
parser.add_argument('--dataset', type=str)
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1 - off)')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--model', type=str, default=None, required=True)
parser.add_argument('--inject_noise', action='store_true')
parser.add_argument('--wd', type=float, default=5e-4)
parser.add_argument('--cold_restarts', action='store_true')
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
        with open(os.path.join(args.dir, 'stdout_cifar_csgmcmc.log'), 'a') as log:
            log.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
sys.stdout = Logger()

device_id = args.device_id
use_cuda = torch.cuda.is_available()

if args.seed != 1:
    print(f'Using given seed {args.seed}')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    
loaders, num_classes = data.loaders(args.dataset, args.data_path, args.batch_size, args.num_workers,
                                    getattr(transforms, args.transform).train, getattr(transforms, args.transform).test)
testloader = loaders['test']
trainloader = loaders['train']
    
net = None

def init_net():
    global net
    arch = getattr(models, args.model)
    net = arch.base(num_classes=num_classes, **arch.kwargs)
    if use_cuda:
        net.cuda(device_id)
        cudnn.benchmark = True
        cudnn.deterministic = True

def prior_loss(prior_std):
    prior_loss = 0.0
    for var in net.parameters():
        nn = torch.div(var, prior_std)
        prior_loss += torch.sum(nn*nn)
    return 0.5*prior_loss

def noise_loss(lr,alpha):
    noise_loss = 0.0
    noise_std = (2/lr*alpha)**0.5  # because we take grad of this term and multiply the result with lr
    for var in net.parameters():
        means = torch.zeros(var.size()).cuda(device_id)
        noise_loss += torch.sum(var * Variable(torch.normal(means, std = noise_std).cuda(device_id),
                           requires_grad = False))
    return noise_loss

def adjust_learning_rate(optimizer, epoch, batch_idx):
    rcounter = epoch*epoch_batches+batch_idx
    cos_inner = np.pi * (rcounter % (T // M))  # pi * <iteration of cycle we are on>
    cos_inner /= T // M  # 0..pi depending on where we are in the cycle
    cos_out = np.cos(cos_inner) + 1
    lr = 0.5*cos_out*args.max_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)

        optimizer.zero_grad()
        lr = adjust_learning_rate(optimizer, epoch,batch_idx)
        outputs = net(inputs)
        if args.inject_noise and (epoch % args.cycle_epochs) + 1 > args.cycle_epochs-args.noise_epochs:
            loss_noise = noise_loss(lr,args.alpha)/len(trainloader)
            loss = criterion(outputs, targets) + loss_noise
        else:
            loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

    print('Train set: Loss %.3f | Acc %.3f%%'
        % (train_loss/(batch_idx+1), 100.*float(correct)/total))

def test(epoch):
    net.eval()
    all_predictions = []
    all_targets = []
    with torch.no_grad():
        test_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(testloader):
            if use_cuda:
                inputs, targets = inputs.cuda(device_id), targets.cuda(device_id)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            
            all_predictions.append(F.softmax(outputs, dim=1).cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    print('Test set:  Loss {:.3f} | Acc {:.3f}% ({}/{})\n'.format(
        test_loss/len(testloader),
        100. * float(correct) / total, correct, total))
    
    all_predictions = np.vstack(all_predictions)
    all_targets = np.concatenate(all_targets).astype('int64')
    return all_predictions, all_targets

prior_std = 1
epoch_batches = len(trainloader)
M = args.cycles
epochs = args.cycle_epochs * args.cycles
T = epochs*epoch_batches # total number of iterations
criterion = nn.CrossEntropyLoss()
optimizer = None
mt = 0

if args.inject_noise:
    method_name = 'cSGLD'
else:
    method_name = 'SSE'
    
    
    
for epoch in range(epochs):
    if epoch == 0 or (args.cold_restarts and epoch%args.cycle_epochs == 0):
        init_net()
        optimizer = optim.SGD(net.parameters(), lr=args.max_lr, momentum=1-args.alpha, weight_decay=args.wd)
    train(epoch)
    preds, targets = test(epoch)
    if (epoch % args.cycle_epochs) + 1 > args.cycle_epochs-args.cycle_saves:
        print('save!')
        net.cpu()
        filename = f'{args.dataset}-{method_name}_{args.model}_run_{args.iter}-{mt}.pt'
        path = os.path.join(args.dir, filename)
        torch.save(net.state_dict(), path)
        mt += 1
        net.cuda(device_id)
