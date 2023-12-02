
"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.sinkhorn_knopp import SinkhornLabelAllocation, SemiCurrSinkhornKnopp
import itertools as it
from termcolor import colored
from losses.ramps import sigmoid_rampup

EPS=1e-8


class SK_loss(nn.Module):
    def __init__(self,p,sk_type="sela", factor=10, num_iter=3, total_iter=100000,start_iter=0,logits_bank=None):
        super(SK_loss, self).__init__()
        sk_iter_limit = p["sk_iter_limit"]
        self.num_heads=p["num_heads"]
        if sk_type=="uot":
            self.sk = [SemiCurrSinkhornKnopp(gamma=p["gamma_bound"], epsilon=factor, numItermax=sk_iter_limit) for _ in range(self.num_heads)]
            p["rho_base"] = 1
            p["rho_upper"] = 1
            p["rho_fix"] = True
            print(colored(f"using uot, rho fixed to: {p['rho_base']}", 'red'))
        elif sk_type=="ppot":
            self.sk = [SemiCurrSinkhornKnopp(gamma=p["gamma_bound"], epsilon=factor, numItermax=sk_iter_limit) for _ in range(self.num_heads)]
        elif sk_type=="pot":
            self.sk = [SemiCurrSinkhornKnopp(gamma=p["gamma_bound"], epsilon=factor,semi_use=False, numItermax=sk_iter_limit) for _ in range(self.num_heads)]
        elif sk_type=="sla":
            self.sk = [SinkhornLabelAllocation(p["num_examples"], p["log_upper_bounds"], allocation_param=0, reg=100, update_tol=1e-2,device="cuda") for _ in range(self.num_heads)]
        else:
            raise NotImplementedError
        self.logits_bank=logits_bank
        self.sk_type=sk_type
        self.criterion=torch.nn.CrossEntropyLoss().cuda()
        self.labels=[[] for _ in range(self.num_heads)] # to compute label acc
        self.target=[] # to compute label acc
        self.i = start_iter
        self.total_iter = total_iter
        self.rho_base=p["rho_base"]
        self.rho_upper = p["rho_upper"] - p["rho_base"]
        self.rho_fix = p["rho_fix"]
        self.rho_strategy = p["rho_strategy"]
        self.label_quality_show = p["label_quality_show"]
        for sk in self.sk:
            sk.rho = p["rho_base"]

    def forward(self, logits, target=None, data_idxs=None):
        # For multi-view: logits[view[head]]
        batch_size=logits[0][0].shape[0]
        if not self.rho_fix:
            self.set_rho(self.i, self.total_iter)
        if self.logits_bank is None:
            if self.sk_type == "sla":
                assert data_idxs is not None, "data_idxs should not be None for SLA"
                pseudo_labels=[[self.sk[head_id](head, data_idxs) for head_id ,head in enumerate(view)] for view in logits]
            else:
                pseudo_labels=[[self.sk[head_id](head) for head_id ,head in enumerate(view)] for view in logits]
        else:
            pseudo_labels=[]
            for view_id,view in enumerate(logits):
                pseudo_labels_view=[]
                for head_id,head in enumerate(view):
                    memory, memory_idx = self.logits_bank[head_id](head,enqueue=True if view_id==0 else False)
                    pseudo_label=self.sk[head_id](memory)[-batch_size:,:] if memory_idx==0 else self.sk[head_id](memory)[memory_idx-batch_size:memory_idx,:]
                    pseudo_labels_view.append(pseudo_label)
                pseudo_labels.append(pseudo_labels_view)
        ### information display
        if self.i % 100 == 0:
            if pseudo_labels[0][0].shape[-1]<=10:
                print(colored(f"The distribution of pseudo_labels: {pseudo_labels[0][0].sum(dim=0).cpu().numpy()}", 'red'))
        ###
        self.i += 1
        total_loss=[]
        for i,(logits_head, label_head) in enumerate(zip(zip(*logits), zip(*pseudo_labels))):
            loss=0
            if self.label_quality_show:
                self.labels[i].append(label_head[0].cpu())
            for a,b in it.permutations(range(len(logits_head)), 2):
                loss+=self.criterion(logits_head[a],label_head[b])

            total_loss.append(loss)
        if target is not None and self.label_quality_show:
            self.target.append(target.cpu())
        return torch.stack(total_loss)

    def single_forward(self, logits):
        pseudo_label = self.sk[0](logits)
        loss=self.criterion(logits,pseudo_label)
        return loss

    def reset(self):
        '''empty the labels and targets recorded'''
        self.labels=[[] for _ in range(self.num_heads)]
        self.target=[]

    def prediction_log(self,top_rho=False):
        assert len(self.target)>0 and len(self.target) == len(self.labels[0])
        probs = [torch.cat(head,dim=0) for head in self.labels]
        predictions = [torch.argmax(head,dim=1) for head in probs]
        targets = torch.cat(self.target,dim=0)
        combine = [{'predictions': pred, 'probabilities': prob, 'targets': targets} for pred,prob in zip(predictions,probs)]

        if top_rho:
            ### get top 10% confidence samples
            select_num = int(targets.size(0) * self.sk[0].rho)
            print(f"top_rho select_num: {select_num}")
            sample_w = [torch.sum(head,dim=1) for head in probs]
            sample_top = [torch.topk(head, select_num, 0, largest=True)[1] for head in sample_w]
            pred_top = [torch.index_select(pred, 0, ind) for pred,ind in zip(predictions, sample_top)]
            prob_top = [torch.index_select(prob, 0, ind) for prob,ind in zip(probs, sample_top)]
            target_top = [torch.index_select(targets, 0, sample) for sample in sample_top]
            combine_top = [{'predictions': pred, 'probabilities': prob, 'targets': target_top[i]} for i,(pred,prob) in enumerate(zip(pred_top, prob_top))]
            ###
            return combine, combine_top
        else:
            return combine

    def set_gamma(self, gamma):
        for sk in self.sk:
            sk.gamma = gamma

    def set_rho(self, current, total):
        # if self.sk_type in ["ppot", "pot", "sla"]:
        for sk in self.sk:
            if self.rho_strategy == "sigmoid":
                sk.rho = sigmoid_rampup(current, total)* self.rho_upper + self.rho_base
            elif self.rho_strategy == "linear":
                sk.rho = current / total * self.rho_upper + self.rho_base
            else:
                raise NotImplementedError
