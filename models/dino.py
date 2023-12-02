"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch

def get_dino_vitb16(grad_from_block=11):
    backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
    for m in backbone.parameters():
        m.requires_grad = False

    # Only finetune layers from block 'args.grad_from_block' onwards
    for name, m in backbone.named_parameters():
        if 'block' in name:
            block_num = int(name.split('.')[1])
            if block_num >= grad_from_block:
                m.requires_grad = True
    backbone.backbone_dim=backbone.embed_dim
    return {"backbone":backbone,"dim":backbone.embed_dim}

def get_parameter_with_grad(model):
    named_params,model_state=model.named_parameters(),model.state_dict()
    state_dict_with_grad={}
    for name, param in named_params:
        if param.requires_grad:
            state_dict_with_grad[name]=model_state[name]
    return state_dict_with_grad