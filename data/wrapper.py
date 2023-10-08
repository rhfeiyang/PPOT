import torch
class two_view_wrapper(torch.utils.data.Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, t):
        to_ret = [self.dataset.__getitem__(t) , self.dataset.__getitem__(t)]

        return to_ret