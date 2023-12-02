"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

"""
This code is based on the Torchvision repository, which was licensed under the BSD 3-Clause.
"""
import os
import pickle
import sys
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
# from utils.mypath import MyPath
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

class CIFAR10(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    base_folder = 'cifar-10-batches-py'
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root="/storage/data/renhui/", train=True, transform=None,
                    download=False):

        super(CIFAR10, self).__init__()
        self.root = root
        self.transform = transform
        self.train = train  # training set or test set
        self.classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        if not check_integrity(path, self.meta['md5']):
            raise RuntimeError('Dataset metadata file not found or corrupted.' +
                               ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            if sys.version_info[0] == 2:
                data = pickle.load(infile)
            else:
                data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            dict: {'image': image, 'target': index of target class, 'meta': dict}
        """
        img, target = self.data[index], self.targets[index]
        img_size = (img.shape[0], img.shape[1])
        img = Image.fromarray(img)
        # class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': img_size, 'index': index}}
        
        return out

    def get_image(self, index):
        img = self.data[index]
        return img
        
    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        for fentry in (self.train_list + self.test_list):
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self):
        if self._check_integrity():
            print('Files already downloaded and verified')
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    def extra_repr(self):
        return "Split: {}".format("Train" if self.train is True else "Test")
class CIFAR100(CIFAR10):
    """`CIFAR100 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.

    This is a subclass of the `CIFAR10` Dataset.
    """

    base_folder = "cifar-100-python"
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    tgz_md5 = "eb9058c3a382ffc7106e4002c42a8d85"
    train_list = [
        ["train", "16019d7e3df5f24257cddd939b257f8d"],
    ]

    test_list = [
        ["test", "f0ef6b0ae62326f3e7ffdfab6717acfc"],
    ]
    meta = {
        "filename": "meta",
        "key": "fine_label_names",
        "md5": "7973b15100ade9c7d40fb424638fde48",
    }
    def __init__(self, root="/storage/data/renhui/", train=True, transform=None,
                    download=False):
        super(CIFAR100, self).__init__(root, train=train,transform=transform,
                                        download=download)
        self.classes = None

from collections import Counter
from utils.mypath import MyPath

def get_imbalance_cifar(num_classes=10,imbalance_ratio=0.1,transform=None,split="train",imb_type='exp'):
    train=split=="train"
    if num_classes==10:
        dataset=CIFAR10(root=MyPath.db_root_dir('cifar10'),train=train,transform=transform,download=True)
    elif num_classes==100:
        dataset=CIFAR100(root=MyPath.db_root_dir('cifar100'),train=train,transform=transform,download=True)
    else:
        raise NotImplementedError

    if train:
        img_num_list = get_img_num_per_cls(dataset.data,num_classes, imb_type, imbalance_ratio)
        dataset.data,dataset.targets = gen_imbalanced_data(dataset.data,dataset.targets,img_num_list)

    print(f"imbalance_ratio:{imbalance_ratio} {Counter(dataset.targets)}")
    print("{} Mode: Contain {} images".format("Train" if train else "Eval", len(dataset.data)))
    return dataset

def _get_class_dict():
    class_dict = dict()
    for i, anno in enumerate(get_annotations()):
        cat_id = anno["category_id"]
        if not cat_id in class_dict:
            class_dict[cat_id] = []
        class_dict[cat_id].append(i)
    return class_dict


def get_img_num_per_cls(data, cls_num, imb_type, imb_factor):
    img_max = len(data) / cls_num
    img_num_per_cls = []
    if imb_type == 'exp':
        for cls_idx in range(cls_num):
            num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
            img_num_per_cls.append(int(num))
    elif imb_type == 'step':
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max))
        for cls_idx in range(cls_num // 2):
            img_num_per_cls.append(int(img_max * imb_factor))
    else:
        img_num_per_cls.extend([int(img_max)] * cls_num)
    return img_num_per_cls

def gen_imbalanced_data(data,targets,img_num_per_cls):
    np.random.seed(0)
    new_data = []
    new_targets = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)

    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        # print(f"idx:{idx}")
        selec_idx = idx[:the_img_num]
        new_data.append(data[selec_idx, ...])
        new_targets.extend([the_class, ] * the_img_num)
    new_data = np.vstack(new_data)
    return new_data, new_targets


def get_annotations(labels):
    annos = []
    for label in labels:
        annos.append({'category_id': int(label)})
    return annos

def get_cls_num_list(cls_num,num_per_cls_dict):
    cls_num_list = []
    for i in range(cls_num):
        cls_num_list.append(num_per_cls_dict[i])
    return cls_num_list

if __name__ == "__main__":
    from data import plot_distribution
    for ratio in [0.05, 0.02, 0.01]:
        dataset = get_imbalance_cifar(100,imbalance_ratio=0.1,split="train")
        nums = np.bincount(dataset.targets)
        plot_distribution(nums,f"CIFAR100 R={int(1/ratio)}",train=True)
