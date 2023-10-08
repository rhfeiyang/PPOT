"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.classifer import Cos_classifier



class ClusteringModel(nn.Module):
    def __init__(self, backbone, nclusters, nheads=1, head_type="linear"):
        assert isinstance(nclusters,list) and (len(nclusters)==1 or len(nclusters)==nheads)
        if len(nclusters)==1 and nheads>1:
            nclusters = [nclusters[0]]*nheads
        super(ClusteringModel, self).__init__()
        self.backbone = backbone['backbone']
        self.backbone_dim = backbone['dim']
        self.nheads = nheads
        assert(isinstance(self.nheads, int))
        assert(self.nheads > 0)
        print(f"head_type: {head_type}")
        self.head_type = head_type
        if head_type=="linear":
            self.cluster_head = nn.ModuleList([nn.Linear(self.backbone_dim, nclusters[i]) for i in range(self.nheads)])
        elif head_type=="cos":
            self.cluster_head = nn.ModuleList([Cos_classifier(self.backbone_dim, nclusters[i]) for i in range(self.nheads)])
        # self.out_dim=nclusters

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = [cluster_head(features) for cluster_head in self.cluster_head]

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = [cluster_head(x) for cluster_head in self.cluster_head]

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out

    def init_prototype(self,prototype):
        assert self.head_type == "cos"
        for cluster_head in self.cluster_head:
            cluster_head.init_prototype(prototype)

