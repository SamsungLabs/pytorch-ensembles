import argparse
import torch
import torch.nn.functional as F
import numpy as np
import os
import tqdm
import sys
from copy import deepcopy

import models
from utils import snapshot_data as data
from utils import snapshot_transforms as transforms
from utils import swag_utils as utils
from utils.swag_model import SWAG

parser = argparse.ArgumentParser(description='SGD/SWA training')
parser.add_argument('--file', type=str, default=None, required=False, help='checkpoint')
parser.add_argument('--dir', type=str, default=None, required=False, help='checkpoint')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--transform', type=str, required=True)
parser.add_argument('--data_path', type=str, default='/scratch/datasets/', metavar='PATH',
                    help='path to datasets location (default: None)')
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--num_workers', type=int, default=4, metavar='N', help='number of workers (default: 4)')
parser.add_argument('--model', type=str, default='VGG16', metavar='MODEL',
                    help='model name (default: VGG16)')
parser.add_argument('--method', type=str, default='SWAG', choices=['SWAG', 'SGD', 'HomoNoise', 'Dropout', 'SWAGDrop'], required=True)
parser.add_argument('--save_path', type=str, default=None, required=False, help='path to npz results file')
parser.add_argument('--N', type=int, default=30)
parser.add_argument('--all_sample_nums', action='store_true')
parser.add_argument('--scale', type=float, default=1.0)
parser.add_argument('--cov_mat', action='store_true', help = 'use sample covariance for swag')
parser.add_argument('--use_diag', action='store_true', help = 'use diag cov for swag')
parser.add_argument('--iter', type=int, required=True)
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--max_num_models', type=int, default=20, help='maximum number of SWAG models to save')
parser.add_argument('--tta', action='store_true', help='test-time augmentation (default: off)')

def nll(outputs, labels):
    labels = labels.astype(int)
    idx = (np.arange(labels.size), labels)
    ps = outputs[idx]
    nll = -np.sum(np.log(ps))
    return nll

args = parser.parse_args()

if args.dir is not None:
    with open(os.path.join(args.dir, f'uncertainty_dir_tta_{args.tta}.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
else:
    dir_ = os.path.dirname(args.file)
    with open(os.path.join(dir_, f'uncertainty_file_tta_{args.tta}.sh'), 'w') as f:
        f.write(' '.join(sys.argv))
        f.write('\n')
        
class Logger():
    def __init__(self):
        self.stdout = sys.stdout  # save it because stdout will be replaced
        
    def write(self, message):
        self.stdout.write(message)
        filename = os.path.join(args.dir, f'stdout_uncertainty_dir_tta_{args.tta}.sh') if args.dir is not None \
            else os.path.join(os.path.dirname(args.file), f'stdout_uncertainty_file_tta_{args.tta}.sh')
        with open(filename, 'a') as log:
            log.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
sys.stdout = Logger()

eps = 1e-12
if args.cov_mat:
    args.cov_mat = True
else:
    args.cov_mat = False

torch.backends.cudnn.benchmark = True
if args.seed != 1:
    print(f'Using given seed {args.seed}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

model_cfg = getattr(models, args.model)

loaders, num_classes = data.loaders(args.dataset, args.data_path, args.batch_size, args.num_workers,
                                    getattr(transforms, args.transform).train, getattr(transforms, args.transform).test)
    
print('Preparing model')
if args.method in ['SWAG', 'HomoNoise', 'SWAGDrop']:
    model = SWAG(model_cfg.base, no_cov_mat=not args.cov_mat, max_num_models=args.max_num_models, *model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
elif args.method in ['SGD', 'Dropout']:
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
else:
    assert False
model.cuda()

def train_dropout(m):
    if type(m)==torch.nn.modules.dropout.Dropout:
        m.train()

if args.file is not None:
    filenames = [args.file]
else:
    filenames = sorted([filename for filename in os.listdir(args.dir) if 'swag_ensembled' in filename and filename.endswith('.pt')],
                       key=lambda fname: int(fname.split('-')[-1].split('.')[0]))
for filename in filenames:
    if args.file is not None:
        full_path = args.file
    else:
        full_path = os.path.join(args.dir, filename)
    print('Loading model %s' % filename)
    checkpoint = torch.load(full_path)
    model.load_state_dict(checkpoint['state_dict'])

    for (module, name) in model.params:
        cov_mat_sqrt = module.__getattr__('%s_cov_mat_sqrt' % name)
        module.__setattr__('%s_cov_mat_sqrt' % name, cov_mat_sqrt.to(torch.device('cpu')))

    if args.method == 'HomoNoise':
        std = 0.01
        for module, name in model.params:
            mean = module.__getattr__('%s_mean' % name)
            module.__getattr__('%s_sq_mean' % name).copy_(mean**2 + std**2)


    predictions = np.zeros((len(loaders['test'].dataset), num_classes))
    targets = np.zeros(len(loaders['test'].dataset))

    for i in range(args.N):
        print('%d/%d' % (i + 1, args.N))

        sample_with_cov = args.cov_mat and not args.use_diag
        model.sample(scale=args.scale, cov=sample_with_cov)
        utils.bn_update(loaders['train'], model)
        model.eval()
    
        filename = f'{args.dataset}-SWAG_{args.model}_run_{args.iter}-{i}.pt'
        path = os.path.join(os.path.dirname(args.file), filename)
        final_dict = deepcopy(model.base.state_dict())
        for key in list(final_dict):
            if ('_mean' in key and 'running' not in key) or '_cov_mat_sqrt' in key:
                del final_dict[key]
        torch.save(final_dict, path)
    
        k = 0
        for input, target in loaders['test']:
            input = input.cuda(non_blocking=True)
            torch.manual_seed(i)

            output = model(input)

            with torch.no_grad():
                cur_preds = output.cpu().numpy()
                cur_targets = target.numpy()

                predictions[k:k+input.size()[0]] += F.softmax(output, dim=1).cpu().numpy()
                targets[k:(k+target.size(0))] = cur_targets
            k += input.size()[0]

        print("Accuracy:", np.mean(np.argmax(predictions, axis=1) == targets))
        print("NLL:", nll(predictions / (i+1), targets) / predictions.shape[0])
    predictions /= args.N
