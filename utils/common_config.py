"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import math
import numpy as np
import torch
import torchvision.transforms as transforms
from utils.collate import collate_custom


def get_model(p, pretrain_path=None):
    # Get backbone
    if p['backbone'] == 'dino_vitb16':
        from models.dino import get_dino_vitb16
        backbone = get_dino_vitb16()
        try:
            p['model_kwargs']['features_dim'] = backbone['dim']
        except:
            pass
    else:
        raise ValueError('Invalid backbone {}'.format(p['backbone']))

    # Setup
    from models.models import ClusteringModel
    model = ClusteringModel(backbone, p['num_classes'], p['num_heads'], head_type=p['head_type'] if 'head_type' in p else 'linear')

    # Load pretrained weights
    if pretrain_path is not None and os.path.exists(pretrain_path):
        state = torch.load(pretrain_path, map_location='cpu')
        missing = model.load_state_dict(state, strict=False)

    elif pretrain_path is not None and not os.path.exists(pretrain_path):
        raise ValueError('Path with pre-trained weights does not exist {}'.format(pretrain_path))

    else:
        pass

    return model


def get_train_dataset(p, transform, to_augmented_dataset=False,
                        split="train"):

    if p['train_db_name'] == 'cifar_im':
        from data.cifar import get_imbalance_cifar
        dataset = get_imbalance_cifar(num_classes=p['num_classes'][0],imbalance_ratio=p['imbalance_ratio'],transform=transform,split=split)

    elif p['train_db_name'] == 'iNature_im':
        from data.inature import get_inaturelist18_datasets
        dataset = get_inaturelist18_datasets(train_transform=transform,split=split,num_classes=p['num_classes'][0])

    elif p['train_db_name'] == 'imagenet-r_im':
        assert p['num_classes'][0] == 200
        from data.imagenet import get_ImageNet_datasets
        dataset = get_ImageNet_datasets(num_classes=p['num_classes'][0],imbalance_factor=p['imbalance_ratio'],transform_train=transform,split=split,version=p['train_db_name'])


    else:
        raise ValueError('Invalid train dataset {}'.format(p['train_db_name']))
    
    # Wrap into other dataset (__getitem__ changes)
    if to_augmented_dataset: # Dataset returns an image and an augmentation of that image.
        from data.custom_dataset import AugmentedDataset
        dataset = AugmentedDataset(dataset)
    
    return dataset

def get_val_dataset(p, transform=None):
    # Base dataset
    if p['val_db_name'] == 'cifar_im':
        from data.cifar import get_imbalance_cifar
        dataset = get_imbalance_cifar(num_classes=p['num_classes'][0],imbalance_ratio=p['imbalance_ratio'],transform=transform,split="val")

    elif p['val_db_name'] == 'iNature_im':
        from data.inature import get_inaturelist18_datasets
        dataset = get_inaturelist18_datasets(test_transform=transform,split="val",num_classes=p['num_classes'][0])

    elif p['val_db_name'] == 'imagenet-r_im':
        from data.imagenet import get_ImageNet_datasets
        dataset = get_ImageNet_datasets(num_classes=p['num_classes'][0],imbalance_factor=p['imbalance_ratio'],transform_val=transform,split="val",version=p['val_db_name'])
    else:
        raise ValueError('Invalid validation dataset {}'.format(p['val_db_name']))

    return dataset


def get_train_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'], 
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=True, shuffle=True)


def get_val_dataloader(p, dataset):
    return torch.utils.data.DataLoader(dataset, num_workers=p['num_workers'],
            batch_size=p['batch_size'], pin_memory=True, collate_fn=collate_custom,
            drop_last=False, shuffle=False)


def get_train_transformations(p):
    if p['backbone'] == 'dino_vitb16':
        from torchvision.transforms import InterpolationMode
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation = InterpolationMode.BICUBIC
        crop_pct = 0.875
        image_size=224
        return transforms.Compose([
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])
    else:
        raise NotImplementedError("unknown backbone")


def get_val_transformations(p):
    if p['backbone'] == 'dino_vitb16':
        from torchvision.transforms import InterpolationMode
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        interpolation=InterpolationMode.BICUBIC
        crop_pct = 0.875
        image_size=224
        return transforms.Compose([
                transforms.Resize(int(image_size / crop_pct), interpolation),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=torch.tensor(mean),
                    std=torch.tensor(std))
            ])
    else:
        raise NotImplementedError("unknown backbone")


def get_optimizer(p, model, cluster_head_only=False):
    if cluster_head_only: # Only weights in the cluster head will be updated 
        for name, param in model.named_parameters():
            print(name)
            if 'cluster_head' in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False 
        params = list(filter(lambda p: p.requires_grad, model.parameters()))
        assert(len(params) == 2 * p['num_heads'])

    else:
        params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if p['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(params, lr=p['lr'], weight_decay=p['weight_decay'])

    elif p['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(params, lr=p['lr'], weight_decay=p['weight_decay'])
    
    else:
        raise ValueError('Invalid optimizer {}'.format(p['optimizer']))

    return optimizer


def adjust_learning_rate(p, optimizer, epoch):
    lr = p['lr']
    
    if p['scheduler'] == 'cosine':
        eta_min = lr * (p['lr_decay_rate'] ** 3)
        lr = eta_min + (lr - eta_min) * (1 + math.cos(math.pi * epoch / p['epochs'])) / 2
         
    elif p['scheduler'] == 'step':
        steps = np.sum(epoch > np.array(p['lr_decay_epochs']))
        if steps > 0:
            lr = lr * (p['lr_decay_rate'] ** steps)

    elif p['scheduler'] == 'constant':
        lr = lr

    else:
        raise ValueError('Invalid learning rate schedule {}'.format(p['scheduler']))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr
