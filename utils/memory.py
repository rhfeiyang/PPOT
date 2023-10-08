"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import numpy as np
import torch


class MemoryBank(object):
    def __init__(self, n, dim, num_classes, temperature):
        self.n = n
        self.dim = dim 
        self.features = torch.FloatTensor(self.n, self.dim)
        self.targets = torch.LongTensor(self.n)
        self.ptr = 0
        self.device = 'cpu'
        self.K = 100
        self.temperature = temperature
        self.C = num_classes

    def weighted_knn(self, predictions):
        # perform weighted knn
        retrieval_one_hot = torch.zeros(self.K, self.C).to(self.device)
        batchSize = predictions.shape[0]
        correlation = torch.matmul(predictions, self.features.t())
        yd, yi = correlation.topk(self.K, dim=1, largest=True, sorted=True)
        candidates = self.targets.view(1,-1).expand(batchSize, -1)
        retrieval = torch.gather(candidates, 1, yi)
        retrieval_one_hot.resize_(batchSize * self.K, self.C).zero_()
        retrieval_one_hot.scatter_(1, retrieval.view(-1, 1), 1)
        yd_transform = yd.clone().div_(self.temperature).exp_()
        probs = torch.sum(torch.mul(retrieval_one_hot.view(batchSize, -1 , self.C), 
                          yd_transform.view(batchSize, -1, 1)), 1)
        _, class_preds = probs.sort(1, True)
        class_pred = class_preds[:, 0]

        return class_pred

    def knn(self, predictions):
        # perform knn
        correlation = torch.matmul(predictions, self.features.t())
        sample_pred = torch.argmax(correlation, dim=1)
        class_pred = torch.index_select(self.targets, 0, sample_pred)
        return class_pred

    def mine_nearest_neighbors(self, topk, calculate_accuracy=True):
        # mine the topk nearest neighbors for every sample
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        index = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_all_gpus(index)
        index.add(features)
        distances, indices = index.search(features, topk+1) # Sample itself is included
        
        # evaluate 
        if calculate_accuracy:
            targets = self.targets.cpu().numpy()
            neighbor_targets = np.take(targets, indices[:,1:], axis=0) # Exclude sample itself for eval
            anchor_targets = np.repeat(targets.reshape(-1,1), topk, axis=1)
            accuracy = np.mean(neighbor_targets == anchor_targets)
            return indices, accuracy
        else:
            return indices

    def get_centroid(self, k):
        # perform k-means clustering
        import faiss
        features = self.features.cpu().numpy()
        n, dim = features.shape[0], features.shape[1]
        kmeans = faiss.Kmeans(dim, k,gpu=True,seed=1, verbose=False, max_points_per_centroid=20000000,nredo=5,niter=60)
        kmeans.train(features)
        # _, indices = kmeans.index.search(features, 1)
        return kmeans.centroids

    def reset(self):
        self.ptr = 0 
        
    def update(self, features, targets):
        b = features.size(0)
        
        assert(b + self.ptr <= self.n)
        
        self.features[self.ptr:self.ptr+b].copy_(features.detach())
        self.targets[self.ptr:self.ptr+b].copy_(targets.detach())
        self.ptr += b

    def to(self, device):
        self.features = self.features.to(device)
        self.targets = self.targets.to(device)
        self.device = device

    def cpu(self):
        self.to('cpu')

    def cuda(self):
        self.to('cuda:0')

class LogitsMemory(torch.nn.Module):
    """Memory Module for NCL"""

    def __init__(self, dim=10, size=5120):
        super(LogitsMemory, self).__init__()
        self.dim = dim  # feature dim
        self.size = size  # memory size
        self.index = 0
        self.is_inited = False
        self.register_buffer('memory', torch.FloatTensor(self.size, dim),persistent=False)
        print('logits bank shape: ({},{})'.format(self.size, dim))
    @torch.no_grad()
    def forward(self, input_logits=None, update=True, enqueue=True):
        if update:
            if not enqueue:
                self.index-=input_logits.size(0)
            idx = self.update_memory(input_logits)
        else:
            idx = self.index
        return self.memory, idx

    @torch.no_grad()
    def update_memory(self, input_logits):
        # update memory
        assert input_logits.size(1) == self.dim
        input_logits = input_logits.detach()
        num = input_logits.size(0)
        out_ids = torch.arange(num).cuda()
        out_ids += self.index
        out_ids %= self.size
        out_ids = out_ids.long()
        self.memory.index_copy_(0, out_ids, input_logits)
        self.index = (self.index + num) % self.size
        return self.index
    @torch.no_grad()
    def init(self, data_loader, model, head=0):
        assert not self.is_inited
        assert self.size<=len(data_loader.dataset)
        # init logits bank, by randomly self.size logits
        model.eval()
        iter_num = self.size // data_loader.batch_size
        for i, batch in enumerate(data_loader):
            batch=batch[0]
            if "image" in batch:
                images = batch['image'].cuda(non_blocking=True)
            elif "anchor" in batch:
                images = batch['anchor'].cuda(non_blocking=True)
            else:
                raise ValueError("image or anchor not in batch")
            logits = model(images)[head]
            self.update_memory(logits)
            if i == iter_num-1:
                break
        self.is_inited = True
        print('logits bank inited')
        return self
