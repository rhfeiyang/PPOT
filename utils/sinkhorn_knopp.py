"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

class SemiCurrSinkhornKnopp(torch.nn.Module):
    """
    naive SinkhornKnopp algorithm for semi-relaxed curriculum optimal transport, one side is equality constraint, the other side is KL divergence constraint (the algorithm is not stable)
    """
    def __init__(self, num_iters=3, epsilon=0.1, gamma=1, stoperr=1e-6, numItermax=1000, rho=0, semi_use=True):
        super().__init__()
        self.num_iters = num_iters
        self.epsilon = epsilon
        self.gamma = gamma
        self.stoperr = stoperr
        self.numItermax = numItermax
        self.rho = rho
        self.b = None
        self.semi_use = semi_use
        print(f"semi_use: {semi_use}")
        print(f"epsilon: {epsilon}")
        print(f"sk_numItermax: {numItermax}")

    @torch.no_grad()
    def forward(self, P):
        # Q is the cost matrix with torchK
        P = P.detach().double()
        P = -torch.log(torch.softmax(P, dim=1))
        n=P.shape[0]
        k=P.shape[1]
        mu = torch.zeros(n, 1).cuda()
        expand_cost = torch.cat([P, mu], dim=1)
        Q = torch.exp(- expand_cost / self.epsilon)
        
        # prior distribution
        Pa = torch.ones(n, 1).cuda() / n  # how many samples
        Pb = self.rho * torch.ones(Q.shape[1], 1).cuda() / k # how many prototypes
        Pb[-1] = 1 - self.rho

        # init b
        b = torch.ones(Q.shape[1], 1).double().cuda() / Q.shape[1] if self.b is None else self.b

        fi = self.gamma / (self.gamma + self.epsilon)
        err = 1
        last_b = b.clone()
        iternum = 0
        while err > self.stoperr and iternum < self.numItermax:
            a = Pa / (Q @ b)
            b =  Pb / (Q.t() @ a)
            if self.semi_use:
                b[:-1,:] = torch.pow(b[:-1,:], fi)

            err = torch.norm(b - last_b)
            last_b = b.clone()
            iternum += 1

        plan = Q.shape[0]*a*Q*b.T
        self.b=b # for two view speed up
        # print(iternum,end=" ")
        # scale the plan
        # plan = plan / torch.sum(plan, dim=1, keepdim=True)
        return plan[:, :-1].float()


class SinkhornLabelAllocation(torch.nn.Module):
    cost_matrix: torch.Tensor
    log_Q: torch.Tensor  # log assignment matrix
    u: torch.Tensor  # row scaling variables
    v: torch.Tensor  # column scaling variables
    log_upper_bounds: torch.Tensor  # log class upper bounds
    rho: float  # allocation fraction
    reg: float  # regularization coefficient
    update_tol: float

    def __init__(
            self,
            num_examples: int,
            log_upper_bounds: torch.Tensor,
            allocation_param: float,
            reg: float,
            update_tol: float,
            device='cpu'):
        super().__init__()
        self.num_examples = num_examples
        self.num_classes = len(log_upper_bounds)
        self.cost_matrix = torch.zeros(self.num_examples + 1, self.num_classes + 1, device=device)
        self.u = torch.zeros(self.num_examples + 1, device=device)
        self.v = torch.zeros(self.num_classes + 1, device=device)
        self.log_upper_bounds = log_upper_bounds
        self.upper_bounds = torch.exp(log_upper_bounds).to(device)
        self.reg = reg
        self.update_tol = update_tol
        self.device = device
        self.set_allocation_param(allocation_param)
        self.reset()

    def set_upper_bounds(self, log_upper_bounds: torch.Tensor):
        self.log_upper_bounds = log_upper_bounds
        self.upper_bounds = torch.exp(log_upper_bounds).to(self.device)

    def reset(self):
        self.u.zero_()
        self.v.zero_()
        self.cost_matrix[:-1, :-1] = np.log(self.num_classes)
        self.log_Q = F.log_softmax(-self.reg * self.cost_matrix, -1)

    def get_plan(self, idxs=None, log_p=None):
        assert (idxs is None or log_p is None)
        if idxs is None and log_p is None:
            return self.log_Q.exp()
        elif idxs is not None:
            return self.log_Q[idxs].exp()
        else:
            z = self.v.repeat(log_p.shape[0], 1)
            z[:, :-1] += self.reg * log_p
            return F.softmax(z, 1)

    def get_assignment(self, idxs=None, log_p=None):
        assert(idxs is None or log_p is None)
        if idxs is None and log_p is None:
            return torch.argmax(self.log_Q[:-1], 1)
        elif idxs is not None:
            return torch.argmax(self.log_Q[idxs], 1)
        else:
            z = self.v.repeat(log_p.shape[0], 1)
            z[:, :-1] += self.reg * log_p
            return torch.argmax(z, 1)

    def set_allocation_param(self, val: float):
        self.rho = val
        return self

    def set_cost_matrix(self, cost_matrix: torch.Tensor):
        self.cost_matrix.copy_(cost_matrix)
        self.log_Q = -self.reg * self.cost_matrix + self.u.view(-1, 1) + self.v.view(1, -1)
        return self

    def update_cost_matrix(self, log_p: torch.Tensor, idxs: torch.LongTensor):
        self.cost_matrix[idxs, :-1] = -log_p.detach()
        log_Q = -self.reg * self.cost_matrix[idxs] + self.v.view(1, -1)
        self.u[idxs] = -torch.logsumexp(log_Q, 1)
        self.log_Q[idxs] = log_Q + self.u[idxs].view(-1, 1)
        return self

    def update(self):
        mat = -self.reg * self.cost_matrix
        iters = 0
        mu = 1 - self.upper_bounds.sum()
        rn = 1 + self.num_classes + self.num_examples * (1 - self.rho - mu.clamp(max=0))
        c = torch.cat([
            1 + self.num_examples * self.upper_bounds,
            1 + self.num_examples * (1 - self.rho + mu.clamp(min=0).view(-1))])

        err = np.inf
        while err >= self.update_tol:
            # update columns
            log_Q = mat + self.u.view(-1, 1)
            self.v = torch.log(c) - torch.logsumexp(log_Q, 0)
            self.v -= self.v[:-1].mean()

            # update rows
            log_Q = mat + self.v.view(1, -1)
            self.u = -torch.logsumexp(log_Q, 1)
            self.u[-1] += torch.log(rn)
            self.log_Q = log_Q + self.u.view(-1, 1)

            err = (torch.abs(self.log_Q.exp().sum(0) - c).sum() / c.sum()).cpu().item()
            iters += 1

        return err, iters
    
    def forward(self, logits, idxs):
        logp_w = F.log_softmax(logits, dim=1)
        q = self.get_plan(log_p=logp_w.detach())
        q = q[:, :-1]
        self.update_cost_matrix(logp_w, idxs.long())
        self.update()
        return q
    

if __name__=="__main__":
    logits = torch.rand(100, 4).cuda()
    pp = SemiCurrSinkhornKnopp(rho=1)(logits)
    print(pp.sum(dim=0), pp.sum(dim=1))
