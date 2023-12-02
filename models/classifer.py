"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import numpy as np
class Cos_classifier(nn.Module):
    def __init__(self, dim, num_prototypes, logit_factor=10.0):
        super().__init__()
        self.logit_factor = logit_factor
        self.embedding = nn.utils.weight_norm(nn.Linear(dim, num_prototypes, bias=False))
        self.embedding.weight_g.data.fill_(1)
        self.embedding.weight_g.requires_grad = False

    def forward(self, x):
        x = torch.nn.functional.normalize(x, dim=1, p=2)
        logits= self.embedding(x)
        return logits * self.logit_factor
    def init_prototype(self,prototype):
        assert prototype.shape==self.embedding.weight_v.data.shape
        self.embedding.weight_v.data.copy_(prototype)

def classify(x, prototypes):
    x = torch.nn.functional.normalize(x, dim=1, p=2)
    head = torch.nn.functional.normalize(prototypes, dim=0, p=2)
    logits = x @ head
    return logits
