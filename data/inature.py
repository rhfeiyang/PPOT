"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import torch
import numpy as np
import torchvision
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
from PIL import Image
import random
import os
import json
from tqdm import tqdm
# from config import iNaturalist18
from copy import deepcopy
import pickle as pkl
from data import plot_distribution
from utils.mypath import MyPath

iNaturalist18 = MyPath.db_root_dir('iNature')
class iNaturalist18Dataset(Dataset):
    def __init__(self, txt, transform=None, target_transform=None, root=iNaturalist18):
        self.samples = []
        self.targets = []

        with open(txt) as f:
            for line in f:
                self.samples.append(os.path.join(root, line.split()[0]))
                self.targets.append(int(line.split()[1]))

        self.transform = transform
        self.target_transform = target_transform
        # self.class_data = [[] for i in range(self.num_classes)]
        # for i in range(len(self.labels)):
        #     y = self.labels[i]
        #     self.class_data[y].append(i)

        # self.cls_num_list = [len(self.class_data[i])
        #                      for i in range(self.num_classes)]
        # sorted_classes = np.argsort(self.cls_num_list)
        # self.class_map = [0 for i in range(self.num_classes)]
        # for i in range(self.num_classes):
        #     self.class_map[sorted_classes[i]] = i

        self.uq_idxs = np.array(range(len(self)))

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, index):
        path = self.samples[index]
        label = self.targets[index]
        with open(path, 'rb') as f:
            sample = Image.open(f).convert('RGB')
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            label = self.target_transform(label)
        # return sample, label, self.uq_idxs[index]
        out={'image':sample,'target':label,'meta':{'index':self.uq_idxs[index]}}
        return out


def subsample_dataset(dataset, idxs):

    mask = np.zeros(len(dataset)).astype('bool')
    mask[idxs] = True

    dataset.samples = np.array(dataset.samples)[mask].tolist()
    dataset.targets = np.array(dataset.targets)[mask].tolist()

    dataset.uq_idxs = dataset.uq_idxs[mask]

    # dataset.samples = [[x[0], int(x[1])] for x in dataset.samples]
    # dataset.targets = [int(x) for x in dataset.targets]

    return dataset


def subsample_classes(dataset, include_classes=range(250)):

    cls_idxs = [
        x for x, l in enumerate(dataset.targets) if l in include_classes
    ]

    target_xform_dict = {}
    for i, k in enumerate(include_classes):
        target_xform_dict[k] = i

    dataset = subsample_dataset(dataset, cls_idxs)

    dataset.target_transform = lambda x: target_xform_dict[x]

    return dataset

# must be desending
all_class_num=[8142 ,2000, 1000, 500, 200, 100]

def recursive_generate(num, all_targets, regenerate=False):
    assert num in all_class_num

    if os.path.exists(f"./data/inature/select_class_{num}.pkl") and not regenerate:
        select_class = torch.load(f"./data/inature/select_class_{num}.pkl")
        return select_class

    idx = all_class_num.index(num)
    if idx == 0:
        targets = torch.IntTensor(all_targets)
        count = torch.bincount(targets)
        select_class = torch.argsort(count, descending=True)
        torch.save(select_class, f"./data/inature/select_class_{num}.pkl")
        return select_class
    else:
        upper_select_class = recursive_generate(all_class_num[idx-1], all_targets,regenerate=regenerate)
        interval = all_class_num[idx-1] // num
        select_class = upper_select_class[::interval][:num]
        torch.save(select_class, f"./data/inature/select_class_{num}.pkl")
        return select_class


def get_inaturelist18_datasets(train_transform=None,
                               test_transform=None,
                                split="val",
                               num_classes=100,
                               seed=0,
                               regenerate=False
                               ):
    np.random.seed(0)
    assert num_classes in all_class_num
    train=split=="train"
    total_classes_num = 8142
    train_txt = os.path.join(iNaturalist18, "iNaturalist18_train.txt")
    val_txt = os.path.join(iNaturalist18, "iNaturalist18_val.txt")

    # print(all_classes[:10])

    if train:
        # Init entire training set
        train_dataset = iNaturalist18Dataset(train_txt, transform=train_transform)
        os.makedirs("./data/inature",exist_ok=True)
        all_classes = recursive_generate(num_classes, train_dataset.targets).tolist()


        # Get labelled training set which has subsampled classes, then subsample some indices from that
        # TODO: Subsampling unlabelled set in uniform random fashion from training data, will contain many instances of dominant class
        train_dataset = subsample_classes(deepcopy(train_dataset),
                                          include_classes=all_classes)

        #### head, medium, tail
        # nums = torch.bincount(torch.IntTensor(list(map(train_dataset.target_transform, train_dataset.targets))))
        # print(f"num class:{num_classes}, imbalance_ratio: {nums[0]/nums[-1]}")
        # plot_distribution(nums, f"iNature{num_classes}")
        ###
        return train_dataset
    else:
        # Get test dataset
        test_dataset = iNaturalist18Dataset(val_txt, transform=test_transform)
        all_classes = torch.load(f"./data/inature/select_class_{num_classes}.pkl").tolist()

        test_dataset = subsample_classes(test_dataset, include_classes=all_classes)

        # nums = torch.bincount(torch.IntTensor(list(map(test_dataset.target_transform, test_dataset.targets))))
        # plot_distribution(nums, f"iNature{num_classes}",train=False)

        return test_dataset

if __name__ == "__main__":
    splits=["train"]
    for s in splits:
        get_inaturelist18_datasets(split=s,num_classes=100,regenerate=False)
        # get_inaturelist18_datasets(split=s,num_classes=200,regenerate=False)
        get_inaturelist18_datasets(split=s,num_classes=500,regenerate=False)
        get_inaturelist18_datasets(split=s,num_classes=1000,regenerate=False)

    # get_inaturelist18_datasets(split="train",num_classes=2000,regenerate=False)
    # root = iNaturalist18
    # json2txt = {
    #     'train2018.json': 'iNaturalist18_train.txt',
    #     'val2018.json': 'iNaturalist18_val.txt'
    # }
    #
    # def convert(json_file, txt_file):
    #     with open(json_file, 'r') as f:
    #         data = json.load(f)
    #
    #     lines = []
    #     for i in tqdm(range(len(data['images']))):
    #         assert data['images'][i]['id'] == data['annotations'][i]['id']
    #         img_name = data['images'][i]['file_name']
    #         label = data['annotations'][i]['category_id']
    #         lines.append(img_name + ' ' + str(label) + '\n')
    #
    #     with open(txt_file, 'w') as ftxt:
    #         ftxt.writelines(lines)
    #
    # for k, v in json2txt.items():
    #     print('===> Converting {} to {}'.format(k, v))
    #     srcfile = os.path.join(root, k)
    #     convert(srcfile, v)
