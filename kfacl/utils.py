import os
import numpy as np
import time
import pandas as pd

import torch.nn.functional as F


def ensemble_predictions(loader, model, num_classes, scale=1.0, num_ens=10, **kwargs):
    predictions = np.zeros((len(loader.dataset), num_classes, num_ens))
    targets = np.zeros(len(loader.dataset))

    for i in range(num_ens):
        model.sample(scale=scale)
        k = 0
        for input, target in loader:
            input = input.cuda(non_blocking=True)
            output = model(input)

            predictions[k:k+input.size()[0], :, i] += F.softmax(output, dim=1).cpu().numpy()
            targets[k:(k+target.size(0))] = target.numpy()
            k += input.size()[0]

    preds = np.mean(predictions[:, :, 0:(i+1)], axis=2)
    trgts = targets.astype(int)
    return preds, trgts


def scale_grid_search(loader, model, num_classes, num_ens=10, scale_range=10**np.linspace(-2, 1, 25)):
    all_losses = np.zeros(scale_range.shape) + np.nan
    for i, scale in list(enumerate(scale_range))[::-1]:
        print("Evaluating %d / %d with scale %.3f" % (i+1, len(scale_range), scale), end='. ')
        t = time.time()
        preds, targets = ensemble_predictions(loader, model, num_classes, scale, num_ens)
        all_losses[i] = np.log(1e-12 + preds[np.arange(len(targets)), targets.astype(int)]).mean().item()
        # all_losses[i] = np.nan
        print("LLH: %.3f. Time %.2f" % (all_losses[i], time.time() - t))
    print('Best LLH: %.3f at scale %.5f' % (np.nanmax(all_losses), scale_range[np.nanargmax(all_losses)]))
    return scale_range[np.nanargmax(all_losses)]


class Logger:
    def __init__(self, base='./logs/'):
        self.res = []
        self.base = base
        os.makedirs(base, exist_ok=True)
        self.df = None

    def add(self, ns, metrics, args, silent=False):
        for m in metrics:
            self.res += [[args.dataset, args.model, args.method, ns, m, metrics[m], '']]
        if not silent:
            print('ns %s: acc %.4f, nll %.4f' % (ns, metrics['acc'], metrics['ll']), flush=True)

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
