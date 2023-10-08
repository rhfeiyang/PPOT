"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import torch
import numpy as np
import errno
from typing import TypeVar, Generator, Union, Sequence, Tuple, Callable, NamedTuple
from scipy.optimize import linear_sum_assignment

def mkdir_if_missing(directory):
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


@torch.no_grad()
def fill_memory_bank(loader, model, memory_bank,forward_pass=None):
    model.eval()
    memory_bank.reset()

    for i, batch in enumerate(loader):
        images = batch['image'].cuda(non_blocking=True)
        if not isinstance(batch['target'], torch.Tensor):
            targets = torch.tensor(batch['target']).cuda(non_blocking=True)
        else:
            targets = batch['target'].cuda(non_blocking=True)
        if forward_pass is None:
            output = model(images)
        else:
            output = model(images,forward_pass=forward_pass)
        if isinstance(output, list):
            output=output[0]
        memory_bank.update(output, targets)
        if i % 100 == 0:
            print('Fill Memory Bank [%d/%d]' %(i, len(loader)))


def confusion_matrix(predictions, gt, class_names, output_file=None,confusion_matrix=None):
    # Plot confusion_matrix and store result to output_file
    import sklearn.metrics
    import matplotlib.pyplot as plt
    if confusion_matrix is None:
        confusion_matrix = sklearn.metrics.confusion_matrix(gt, predictions)
    # confusion_matrix = confusion_matrix / np.sum(confusion_matrix, 1,keepdims=True)
    
    fig, axes = plt.subplots(1)
    plt.imshow(confusion_matrix / np.sum(confusion_matrix, 1,keepdims=True), cmap='Blues')
    if class_names is not None:
        axes.set_xticks([i for i in range(len(class_names))])
        axes.set_yticks([i for i in range(len(class_names))])
        axes.set_xticklabels(class_names, ha='right', fontsize=8, rotation=40)
        axes.set_yticklabels(class_names, ha='right', fontsize=8)

    if confusion_matrix.shape[0] <= 10:
        for (i, j), z in np.ndenumerate(confusion_matrix):
            if i == j or z>0.0:
                axes.text(j, i, '%d' %(z), ha='center', va='center', color='white', fontsize=6)
            else:
                pass

    plt.tight_layout()
    if output_file is None:
        plt.show()
    else:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


class PiecewiseLinear(NamedTuple):
    knots: Sequence[float]
    vals: Sequence[float]

    def __call__(self, t):
        return np.interp([t], self.knots, self.vals)[0]

@torch.no_grad()
def _hungarian_match(flat_preds, flat_targets, preds_k, targets_k):
    # Based on implementation from IIC
    num_samples = flat_targets.shape[0]

    assert (preds_k == targets_k)  # one to one
    num_k = preds_k
    num_correct = np.zeros((num_k, num_k))

    for c1 in range(num_k):
        for c2 in range(num_k):
            # elementwise, so each sample contributes once
            votes = int(((flat_preds == c1) * (flat_targets == c2)).sum())
            num_correct[c1, c2] = votes

    # num_correct is small
    match = linear_sum_assignment(num_samples - num_correct)
    match = np.array(list(zip(*match)))

    # return as list of tuples, out_c to gt_c
    res = []
    for out_c, gt_c in match:
        res.append((out_c, gt_c))

    return res