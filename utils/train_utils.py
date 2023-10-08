"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
from utils.utils import AverageMeter, ProgressMeter
import torch.nn.functional as F


def sk_train(train_loader, model, criterion, optimizer, epoch, sk_w=1.0):
    """
    Train sk-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    sk_losses = AverageMeter('SK Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [total_losses, sk_losses],
                             prefix="Epoch: [{}]".format(epoch))
    all_losses=[] # loss for each head
    model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        imgs=[b["image"].cuda(non_blocking=True) for b in batch]
        target = torch.Tensor(batch[0]['target']) if not isinstance(batch[0]['target'], torch.Tensor) else batch[0]['target']
        logits=[]

        for img in imgs:
            img_output = model(img)
            logits.append(img_output)
            # pseudo_labels.append([optimize_L_sk(head) for head in img_output])

        total_loss=0
        # Sinkhorn loss
        # sk_loss = torch.sum(criterion_sk(logits,target=target))

        sk_loss = criterion(logits, target=target, data_idxs=batch[0]['meta']['index']) # multi-head
        total_loss += sk_w * sk_loss
        sk_losses.update(sk_loss.sum().item())


        # Register the mean loss and backprop the total loss to cover all subheads
        all_losses.append(total_loss.detach())
        total_loss=total_loss.sum()
        total_losses.update(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            progress.display(i)

    ave_loss=torch.stack(all_losses).mean(dim=0)
    lowest_loss_head=ave_loss.argmin().item()
    lowest_loss=ave_loss[lowest_loss_head].item()
    return {'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss, 'total_loss':ave_loss.cpu().numpy()}


def sla_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        imgs=[b["image"].cuda(non_blocking=True) for b in batch]
        features=[]
        logits=[]

        for img in imgs:
            out=model(img,forward_pass="return_all")
            img_features=out['features']
            features.append(img_features.detach())
            img_output = out['output']
            logits.append(img_output)
            # pseudo_labels.append([optimize_L_sk(head) for head in img_output])

        loss = criterion(logits, target, i, batch['meta']['index']) # multi-head
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)


def selflabel_sla_train(train_loader, model, criterion, optimizer, epoch, ema=None):
    """ 
    Self-labeling based on confident samples
    """
    losses = AverageMeter('Loss', ':.4e')
    progress = ProgressMeter(len(train_loader), [losses],
                                prefix="Epoch: [{}]".format(epoch))
    model.train()

    for i, batch in enumerate(train_loader):
        images = batch['image'].cuda(non_blocking=True)
        images_augmented = batch['image_augmented'].cuda(non_blocking=True)

        with torch.no_grad(): 
            output = model(images)[0]
        output_augmented = model(images_augmented)[0]
        loss = criterion(output, output_augmented, i, batch['meta']['index'])
        losses.update(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if ema is not None: # Apply EMA to update the weights of the network
            ema.update_params(model)
            ema.apply_shadow(model)
        
        if i % 25 == 0:
            progress.display(i)

# from losses.IID_losses import IID_loss
def two_view_train(train_loader, model, criterion, optimizer, epoch, update_cluster_head_only=False):
    """
    Train w/ IIC-Loss
    """
    total_losses = AverageMeter('Total Loss', ':.4e')
    criterion_losses = AverageMeter('Criterion Loss', ':.4e')
    progress = ProgressMeter(len(train_loader),
                             [total_losses, criterion_losses],
                             prefix="Epoch: [{}]".format(epoch))
    all_losses=[] # loss for each head
    if update_cluster_head_only:
        model.eval() # No need to update BN
    else:
        model.train() # Update BN

    for i, batch in enumerate(train_loader):
        # Forward pass
        imgs=[b["image"].cuda(non_blocking=True) for b in batch]
        target = torch.Tensor(batch[0]['target']) if not isinstance(batch[0]['target'], torch.Tensor) else batch[0]['target']
        features=[]
        logits=[]

        for img in imgs:
            if update_cluster_head_only: # Only calculate gradient for backprop of linear layer
                with torch.no_grad():
                    img_features = model(img, forward_pass='backbone')
                    features.append(img_features.detach())
                    # neighbor_features = model(neighbor, forward_pass='backbone')
                img_output = model(img_features, forward_pass='head')
                logits.append(img_output)
                # pseudo_labels.append([optimize_L_sk(head) for head in img_output])
                # neighbor_logit = model(neighbor_features, forward_pass='head')
            else: # Calculate gradient for backprop of complete network
                out=model(img,forward_pass="return_all")
                img_features=out['features']
                features.append(img_features.detach())
                img_output = out['output']
                logits.append(img_output)
                # pseudo_labels.append([optimize_L_sk(head) for head in img_output])

        total_loss=0
        criterion_loss=[]
        for view1_head, view2_head in zip(*logits):
            criterion_loss.append(criterion(view1_head, view2_head))
        criterion_loss=torch.stack(criterion_loss)
        total_loss+=criterion_loss.sum()
        criterion_losses.update(total_loss.item())

        # Register the mean loss and backprop the total loss to cover all subheads
        # all_losses.append(total_loss.detach())
        total_losses.update(total_loss.item())

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if i % 10 == 0:
            progress.display(i)
        # return

    # ave_loss=torch.stack(all_losses).mean(dim=0)
    # lowest_loss_head=ave_loss.argmin().item()
    # lowest_loss=ave_loss[lowest_loss_head].item()
    # return {'lowest_loss_head': lowest_loss_head, 'lowest_loss': lowest_loss}
