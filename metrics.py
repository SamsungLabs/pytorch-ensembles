from sklearn.metrics import roc_auc_score as auroc, average_precision_score as prauc
import numpy as np
import torch
from torch.nn.functional import log_softmax
from sklearn.model_selection import KFold
from collections import defaultdict
from scipy.optimize import minimize


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def _misclass_tgt(output, target, topk):
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].float().sum(0)
            res.append(correct_k)

        return res[0].numpy()

def get_acc(preds, targets, **args):
    return np.mean(np.argmax(preds, axis=1) == targets)

def acc_aac(preds, targets, steps=1000, return_plot=False, **args):
    idx = np.argsort(preds.max(1))
    preds_, targets_ = np.argmax(preds[idx], 1), targets[idx]
    step = int(len(preds_)/steps)

    accs = []
    for i in range(1, len(preds_), step):
        accs += [np.mean(targets_[i:] == preds_[i:])]

    accs = np.array(accs)

    if return_plot:
        return accs, 1-np.trapz(accs)/steps

    return 1-np.trapz(accs)/steps

def get_ll(preds, targets, **args):
    return np.log(1e-12 + preds[np.arange(len(targets)), targets]).mean()

def get_acc5(preds, targets, **args):
    preds = torch.Tensor(preds)
    targets = torch.LongTensor(targets)
    return accuracy(preds, targets, topk=(5,))[0].item()/100.

def misclass_tgt(preds, targets, topk, **args):
    preds = torch.Tensor(preds)
    targets = torch.LongTensor(targets)
    return _misclass_tgt(preds, targets, topk=(topk,))

def get_ece(preds, targets, n_bins=15, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences, predictions = np.max(preds, 1), np.argmax(preds, 1)
    accuracies = (predictions == targets)
    
    ece = 0.0
    avg_confs_in_bins = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = np.logical_and(confidences > bin_lower, confidences <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(accuracies[in_bin])
            avg_confidence_in_bin = np.mean(confidences[in_bin])
            delta = avg_confidence_in_bin - accuracy_in_bin
            avg_confs_in_bins.append(delta)
            ece += np.abs(delta) * prop_in_bin
        else:
            avg_confs_in_bins.append(None)
    # For reliability diagrams, also need to return these:
    # return ece, bin_lowers, avg_confs_in_bins
    return ece

def get_sce(preds, targets, n_bins=15, **args):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    n_objects, n_classes = preds.shape
    res = 0.0
    for cur_class in range(n_classes):
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            cur_class_conf = preds[:, cur_class]
            in_bin = np.logical_and(cur_class_conf > bin_lower, cur_class_conf <= bin_upper)

            # cur_class_acc is ground truth probability of chosen class being the correct one inside the bin.
            # NOT fraction of correct predictions in the bin
            # because it is compared with predicted probability
            bin_acc = (targets[in_bin] == cur_class)
            
            bin_conf = cur_class_conf[in_bin]

            bin_size = np.sum(in_bin)
            
            if bin_size > 0:
                avg_confidence_in_bin = np.mean(bin_conf)
                avg_accuracy_in_bin = np.mean(bin_acc)
                delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#                 print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
                res += delta * bin_size / (n_objects * n_classes)
    return res

def get_tace(preds, targets, n_bins=15, threshold=1e-3, **args):
    n_objects, n_classes = preds.shape
    
    res = 0.0
    for cur_class in range(n_classes):
        cur_class_conf = preds[:, cur_class]
        
        targets_sorted = targets[cur_class_conf.argsort()]
        cur_class_conf_sorted = np.sort(cur_class_conf)
        
        targets_sorted = targets_sorted[cur_class_conf_sorted > threshold]
        cur_class_conf_sorted = cur_class_conf_sorted[cur_class_conf_sorted > threshold]
        
        bin_size = len(cur_class_conf_sorted) // n_bins
                
        for bin_i in range(n_bins):
            bin_start_ind = bin_i * bin_size
            if bin_i < n_bins-1:
                bin_end_ind = bin_start_ind + bin_size
            else:
                bin_end_ind = len(targets_sorted)
                bin_size = bin_end_ind - bin_start_ind  # extend last bin until the end of prediction array
            bin_acc = (targets_sorted[bin_start_ind : bin_end_ind] == cur_class)
            bin_conf = cur_class_conf_sorted[bin_start_ind : bin_end_ind]
            avg_confidence_in_bin = np.mean(bin_conf)
            avg_accuracy_in_bin = np.mean(bin_acc)
            delta = np.abs(avg_confidence_in_bin - avg_accuracy_in_bin)
#             print(f'bin size {bin_size}, bin conf {avg_confidence_in_bin}, bin acc {avg_accuracy_in_bin}')
            res += delta * bin_size / (n_objects * n_classes)
            
    return res

def get_ace(preds, targets, n_bins=15, **args):
    return get_tace(preds, targets, n_bins, threshold=0)

def get_brier(preds, targets, **args):
    one_hot_targets = np.zeros(preds.shape)
    one_hot_targets[np.arange(len(targets)), targets] = 1.0
    return np.mean(np.sum((preds - one_hot_targets) ** 2, axis=1))

def get_misclass_auroc(preds, targets, criterion, topk=1, **args):
    misclassification_targets = (1-misclass_tgt(preds, targets, topk)).astype(bool)

    if criterion == 'entropy':
        criterion_values = np.sum(-preds * np.log(preds), axis=1)
    elif criterion == 'confidence':
        criterion_values = -preds.max(axis=1)
    elif criterion == 'MI':
        criterion_values = np.sum(-preds * np.log(preds), axis=1) - args['mean_ens_entropy']
    else:
        raise NotImplementedError

    return auroc(misclassification_targets, criterion_values)

def get_misclass_aucpr(preds, targets, criterion, topk=1, **args):
    misclassification_targets = (1-misclass_tgt(preds, targets, topk)).astype(bool)

    if criterion == 'entropy':
        criterion_values = np.sum(-preds * np.log(preds), axis=1)
    elif criterion == 'confidence':
        criterion_values = -preds.max(axis=1)
    elif criterion == 'MI':
        criterion_values = np.sum(-preds * np.log(preds), axis=1) - args['mean_ens_entropy']
    else:
        raise NotImplementedError

    return prauc(misclassification_targets, criterion_values)

def compute_test_metrics(preds, targets, **args):
    metric_name_to_f = {
        'acc': get_acc,
        'll': get_ll,
        'brier': get_brier,
        'acc_aac': acc_aac,
    }
    
    res = {}
    for metric, f in metric_name_to_f.items():
        res[metric] = f(preds, targets, **args)
    return res

def apply_t(log_preds, t):
     return log_softmax(torch.Tensor(log_preds / t), dim=1).data.numpy()

def ts(log_preds, targets):
    f = lambda t: -get_ll(np.exp(apply_t(log_preds, t)), targets)
    res = minimize(f, 1, method='nelder-mead', options={'xtol': 1e-3})
    return res.x[0]

def metrics_kfold(
        log_preds, targets, n_splits=2, n_runs=5, verbose=False, temp_scale=False, **args):
    metrics = defaultdict(lambda: 0.0)
    for runs in range(n_runs):
        for i, (tr_idx, te_idx) in enumerate(KFold(n_splits=n_splits, shuffle=True).split(log_preds)):
            if temp_scale:
                train_t = ts(log_preds[tr_idx], targets[tr_idx])
                test_lp = apply_t(log_preds[te_idx], train_t)
            else:
                test_lp = log_preds[te_idx]
            te_metrics = compute_test_metrics(np.exp(test_lp), targets[te_idx], **args)

            for k, v in te_metrics.items():
                metrics[k] += v/(n_splits*n_runs)

    if verbose:
        print(metrics)

    return dict(metrics)
