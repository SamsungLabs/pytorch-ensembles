import os
import numpy as np
from collections import defaultdict
from glob import glob
import pandas as pd
from random import shuffle
import torch
from scipy.special import softmax
from functools import partial

from metrics import metrics_kfold

    
def is_probs(data):
    return np.allclose(data.sum(axis=1), np.ones((data.shape[0],)), atol=1e-2, rtol=0)
    
def ensure_probs(data):
    """Accepts either logits or probabilities, outputs probabilities
    data shape: n_objects x n_classes
    """
    if not is_probs(data):
        # logits detected, convert to probs
        return softmax(data, axis=1)
    return data

def ensure_logits(data):
    """Accepts either logits or probabilities, outputs logits
    data shape: n_objects x n_classes
    """
    if is_probs(data):
        return np.log(data)
    return data

def get_nets_filenames(base_path, sub_path, npz_wildcard, npz_filter, npz_sort_key, npz_count, shuffle_npzs):
    fnames = glob(os.path.join(base_path, sub_path, npz_wildcard))
    if npz_filter is not None:
        fnames = list(filter(npz_filter, fnames))
    if not shuffle_npzs:
        fnames = list(sorted(fnames, key=npz_sort_key))
    else:
        shuffle(fnames)
    if npz_count is not None:
        fnames = fnames[:npz_count]
    if len(fnames) < 2:
        return fnames, None, None
    if not shuffle_npzs:
        key0 = npz_sort_key(fnames[0])
        key1 = npz_sort_key(fnames[1])
        if type(key0) == tuple:
            key0 = key0[0]
            key1 = key1[0]
        start_ind = key0
        cycle_epochs = key1 - start_ind
        return fnames, start_ind, cycle_epochs
    else:
        return fnames, None, None
    
def compute_metrics_on_filenames(fnames, start_ind, cycle_epochs, method, arch, dataset, res, base_path, sub_path, ensemble_on_the_go,
                                 npz_sort_key=None, shuffle_npzs=False,
                                 temp_scale=False, save_ens_preds_sizes=[]):
    if len(fnames) < 2:
        return res
    if res is None:
        res = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))))
    if ensemble_on_the_go:
        ens_preds = None
        mean_ens_entropy = None
    get_metrics_f = partial(metrics_kfold, temp_scale=temp_scale,
	disable=('misclass_MI_auroc', 'sce', 'ace', 'misclass_entropy_auroc@5',
		 'misclass_confidence_auroc@5', 'tace', 'misclass_entropy_prauc',
		 'misclass_confidence_prauc'))
    
    for i, fname in enumerate(fnames):
        if not shuffle_npzs:
            key = npz_sort_key(fname)
            if type(key) == tuple:
                cur_ind = i
            else:
                cur_ind = (key - start_ind) // cycle_epochs + 1  # starting from 1
        else:
            cur_ind = i+1
            
        f = np.load(fname)
        npz_preds = f['test_preds']
        npz_targets = f['test_targets']
        npz_preds = ensure_probs(npz_preds)
        mean_npz_entropy = np.sum(-npz_preds * np.log(npz_preds), axis=1)
        
        if ensemble_on_the_go:
            ens_preds = (ens_preds * i + npz_preds) / (i+1) if ens_preds is not None else npz_preds
            mean_ens_entropy = (mean_ens_entropy * i + mean_npz_entropy) / (i+1) if mean_ens_entropy is not None else mean_npz_entropy
            cur_res = get_metrics_f(ensure_logits(ens_preds), npz_targets, mean_ens_entropy=mean_ens_entropy)
            
            if i+1 in save_ens_preds_sizes:
                np.savez(os.path.join(base_path, sub_path, f"ens_preds_{i+1}_members.npz"),
                         test_preds=ens_preds, test_targets=npz_targets)
        else:
            cur_res = get_metrics_f(ensure_logits(npz_preds), npz_targets)
        for metric, value in cur_res.items():
            res[method][metric][dataset][arch][cur_ind].append(value)
    return res

def compute_metrics(method, arch, dataset, res, base_path, sub_path, npz_wildcard, ensemble_on_the_go,
                    npz_filter=None, npz_sort_key=None, npz_count=None, shuffle_npzs=False,
                    temp_scale=False, save_ens_preds_sizes=[]):
    fnames, start_ind, cycle_epochs = get_nets_filenames(base_path, sub_path, npz_wildcard, npz_filter, npz_sort_key, npz_count, shuffle_npzs)
    return compute_metrics_on_filenames(fnames, start_ind, cycle_epochs, method, arch, dataset, res, base_path, sub_path, ensemble_on_the_go,
                                        npz_sort_key, shuffle_npzs, temp_scale, save_ens_preds_sizes)

def res_to_lists(res):
    csv_res = defaultdict(list)
    for method, method_res in res.items():
        for metric, metric_res in method_res.items():
            for dataset, dataset_res in metric_res.items():
                for arch, arch_res in dataset_res.items():
                    for num, num_res in arch_res.items():
                        for val in num_res:
                            csv_res[(dataset, arch, method)].append([dataset, arch, method, num, metric, val, ''])
    return csv_res

def res_to_csv(res, csv_dir, tag):
    csv_res = res_to_lists(res)
    os.makedirs(csv_dir, exist_ok=True)
    for (dataset, arch, method), cur_res in csv_res.items():
        cols = ['dataset', 'model', 'method', 'n_samples', 'metric', 'value', 'info']
        df = pd.DataFrame(cur_res, columns=cols)
        if len(tag) > 0:
            df.to_csv(os.path.join(csv_dir, f'{dataset}-{arch}-{method}-{tag}.csv'))
        else:
            df.to_csv(os.path.join(csv_dir, f'{dataset}-{arch}-{method}.csv'))
