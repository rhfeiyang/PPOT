"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch
import torch.nn.functional as F

from utils.utils import AverageMeter, confusion_matrix,_hungarian_match

from sklearn import metrics


# from termcolor import colored

@torch.no_grad()
def contrastive_evaluate(val_loader, model, memory_bank):
    top1 = AverageMeter('Acc@1', ':6.2f')
    model.eval()

    for batch in val_loader:
        images = batch['image'].cuda(non_blocking=True)
        target = batch['target'].cuda(non_blocking=True)

        output = model(images)
        output = memory_bank.weighted_knn(output) 

        acc1 = 100*torch.mean(torch.eq(output, target).float())
        top1.update(acc1.item(), images.size(0))

    return top1.avg

@torch.no_grad()
def get_predictions(p, dataloader, model):
    # Make predictions on a dataset with neighbors
    model.eval()
    predictions = [[] for _ in range(p['num_heads'])]
    probs = [[] for _ in range(p['num_heads'])]
    logits = [[] for _ in range(p['num_heads'])]
    targets = []

    key_ = 'image'

    ptr = 0
    for batch in dataloader:
        if isinstance(batch,list):
            batch=batch[0]
        if not isinstance(batch['target'],torch.Tensor):
            batch['target']=torch.tensor(batch['target'])
        images = batch[key_].cuda(non_blocking=True)
        bs = images.shape[0]
        res = model(images, forward_pass='return_all')
        output = res['output']
        for i, output_i in enumerate(output):
            predictions[i].append(torch.argmax(output_i, dim=1))
            probs[i].append(F.softmax(output_i, dim=1))
            logits[i].append(output_i)
        targets.append(batch['target'])

    predictions = [torch.cat(pred_, dim = 0) for pred_ in predictions]
    probs = [torch.cat(prob_, dim=0) for prob_ in probs]
    logits = [torch.cat(logit_, dim=0) for logit_ in logits]
    targets = torch.cat(targets, dim=0)

    out = [{'predictions': pred_, 'probabilities': prob_, "logits": logits[i], 'targets': targets} for i,(pred_, prob_) in enumerate(zip(predictions, probs))]
    return out

@torch.no_grad()
def hungarian_evaluate(subhead_index, all_predictions, num_classes=None, class_names=None,
                        compute_purity=True, compute_confusion_matrix=True,
                        confusion_matrix_file=None):
    # Evaluate model based on hungarian matching between predicted cluster assignment and gt classes.
    # This is computed only for the passed subhead index.

    # Hungarian matching
    head = all_predictions[subhead_index]
    targets = head['targets'].cuda()
    predictions = head['predictions'].cuda()
    # probs = head['probabilities'].cuda()
    num_classes = torch.unique(targets).numel() if num_classes is None else num_classes
    num_elems = targets.size(0)

    match = _hungarian_match(predictions, targets, preds_k=num_classes, targets_k=num_classes)
    reordered_preds = torch.zeros(num_elems, dtype=predictions.dtype).cuda()
    for pred_i, target_i in match:
        reordered_preds[predictions == int(pred_i)] = int(target_i)

    # Gather performance metrics
    target_cpu=targets.cpu().numpy()
    predictions_cpu=predictions.cpu().numpy()
    reordered_preds_cpu=reordered_preds.cpu().numpy()
    # acc = int((reordered_preds == targets).sum()) / float(num_elems)
    nmi = metrics.normalized_mutual_info_score(target_cpu, predictions_cpu)
    ari = metrics.adjusted_rand_score(target_cpu, predictions_cpu)
    ###
    confusion = metrics.confusion_matrix(target_cpu, reordered_preds_cpu)
    with np.errstate(divide="ignore", invalid="ignore"):
        class_acc = np.diag(confusion) / np.sum(confusion, axis=1)
    if np.any(np.isnan(class_acc)):
        class_acc = class_acc[~np.isnan(class_acc)]
    balance_acc=class_acc.mean()
    class_f1 = metrics.f1_score(target_cpu, reordered_preds_cpu,average=None,zero_division=0)
    f1_score = class_f1.mean()
    # class_precision=metrics.precision_score(target_cpu, reordered_preds_cpu,average=None,zero_division=0)
    # precision=class_precision.mean()
    ###

    # _, preds_top5 = probs.topk(5, 1, largest=True)
    # reordered_preds_top5 = torch.zeros_like(preds_top5)
    # for pred_i, target_i in match:
    #     reordered_preds_top5[preds_top5 == int(pred_i)] = int(target_i)
    # correct_top5_binary = reordered_preds_top5.eq(targets.view(-1,1).expand_as(reordered_preds_top5))
    # top5 = float(correct_top5_binary.sum()) / float(num_elems)

    # Compute confusion matrix
    if compute_confusion_matrix:
        confusion_matrix(reordered_preds_cpu, target_cpu,
                            class_names, confusion_matrix_file, confusion_matrix=confusion)
    class_acc = [round(i*100,2) for i in class_acc]
    # class_precision = [round(i*100,2) for i in class_precision]
    class_f1 = [round(i*100,2) for i in class_f1]
    acc_analyze = class_acc_analyze(class_acc)
    return {'mACC': round(balance_acc*100,2),'F1': round(f1_score*100,2),'ARI': round(ari*100,2), 'NMI': round(nmi*100,2),
            'class_acc': class_acc,"ACC_head_medium_tail": acc_analyze, 'class_f1': class_f1 , 'hungarian_match': match if len(match) <= 10 else "omit"}

def class_acc_analyze(class_acc):
    num = len(class_acc)
    class_acc = np.array(class_acc)
    head_num = int(num*0.3)
    tail_num = int(num*0.3)
    medium_num = int(num*0.4)
    head_acc = round(class_acc[:head_num].mean(),2)
    tail_acc = round(class_acc[-tail_num:].mean(),2)
    medium_acc = round(class_acc[head_num:head_num+medium_num].mean(),2)
    return head_acc,medium_acc,tail_acc

