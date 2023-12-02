"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
# 1. Generate data split
# 2. Supervised Pretrain dataloader
# 3. Discovery dataloader
import torch
import numpy as np
import os
import pickle as pkl
# from data.data_utils import subsample_instances
# from config import imagenet_root
import torch.utils.data as data
from PIL import Image
import glob
from torchvision import get_image_backend
from utils.mypath import MyPath

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


# def accimage_loader(path):
#     import accimage
#     try:
#         return accimage.Image(path)
#     except IOError:
#         # Potentially a decoding problem, fall back to PIL.Image
#         return pil_loader(path)


def default_loader(path):
    return pil_loader(path)
    # if get_image_backend() == 'accimage':
    #     return accimage_loader(path)
    # else:
    #     return pil_loader(path)


class ImageNetDataset(data.Dataset):

    def __init__(self, root, anno_file, loader=default_loader, transform=None, target_transform=None):
        self.root = root
        self.loader = loader
        self.transform = transform
        self.target_transform = target_transform

        filenames = []
        targets = []
        with open(anno_file, 'r') as fin:
            for line in fin.readlines():
                line_split = line.strip('\n').split(' ')
                filenames.append(line_split[0])
                targets.append(int(line_split[1]))

        self.samples = filenames
        self.targets = targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path = os.path.join(self.root, self.samples[index])
        target = self.targets[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        out = {'image': sample, 'target': target, 'meta': {'index': index}}

        return out
        # return sample, target, index

    def __len__(self):
        return len(self.targets)

imagenet_root = {"imagenet10":MyPath.db_root_dir('imagenet10'),
                 "imagenet100":MyPath.db_root_dir('imagenet100'),
                 "imagenet-r":MyPath.db_root_dir('imagenet-r')}
def get_ImageNet_datasets(num_classes,imbalance_factor=0.1,transform_train=None, transform_val=None,regenerate=False,split="train",version="imagenet-r"):
    train=split=="train"
    version = version.split("_")[0]
    if version=="imagenet":
        version=f"imagenet{num_classes}"
    if version=="imagenet-r":
        train_data_dir = imagenet_root[version]
        val_data_dir = imagenet_root[version]
        train_anno_file = f'./data/imagenet/{version}/{version}_label_{num_classes}.txt'
    else:
        train_data_dir = os.path.join(imagenet_root[version], "ImgNetTrain")
        val_data_dir = os.path.join(imagenet_root[version], "ImgNetVal")
        train_anno_file = f'./data/imagenet/{version}/{version}_label_{num_classes}_imbalance_{imbalance_factor}.txt'

    val_anno_file = f'./data/imagenet/{version}/{version}_numclass_{num_classes}_val.txt'
    if regenerate or not os.path.exists(train_anno_file):
        nums = generate(num_classes, imbalance_factor,version=version)

     # else:
     #     with open(f"./data/imagenet/{version}/{version}_numclass_{num_classes}_imbalance_{imbalance_factor}.pkl",'rb') as f:
     #         nums = pkl.load(f)
     #     from data import plot_distribution
     #     plot_distribution(nums, f"{version} imbalance:{imbalance_factor}",train=train)

    if train:
        train_dataset = ImageNetDataset(train_data_dir, train_anno_file, transform=transform_train)
        return train_dataset
    else:
        test_dataset = ImageNetDataset(val_data_dir, val_anno_file, transform=transform_val)
        return test_dataset


folder_set={"imagenet10": "n02056570 n02085936 n02128757 n02690373 n02692877 n03095699 n04254680 n04285008 n04467665 n07747607",
         "imagenet100": 'n01558993 n01601694 n01669191 n01751748 n01755581 n01756291 n01770393 n01855672 n01871265 n02018207 ' \
                        'n02037110 n02058221 n02087046 n02088632 n02093256 n02093754 n02094114 n02096177 n02097130 n02097298 ' \
                        'n02099267 n02100877 n02104365 n02105855 n02106030 n02106166 n02107142 n02110341 n02114855 n02120079 ' \
                        'n02120505 n02125311 n02128385 n02133161 n02277742 n02325366 n02364673 n02484975 n02489166 n02708093 ' \
                        'n02747177 n02835271 n02906734 n02909870 n03085013 n03124170 n03127747 n03160309 n03255030 n03272010 ' \
                        'n03291819 n03337140 n03450230 n03483316 n03498962 n03530642 n03623198 n03649909 n03710721 n03717622 ' \
                        'n03733281 n03759954 n03775071 n03814639 n03837869 n03838899 n03854065 n03929855 n03930313 n03954731 ' \
                        'n03956157 n03983396 n04004767 n04026417 n04065272 n04200800 n04209239 n04235860 n04311004 n04325704 ' \
                        'n04336792 n04346328 n04380533 n04428191 n04443257 n04458633 n04483307 n04509417 n04515003 n04525305 ' \
                        'n04554684 n04591157 n04592741 n04606251 n07583066 n07613480 n07693725 n07711569 n07753592 n11879895',}
def generate(num_classes,imbalance_factor, version="imagenet10"):
    if version=="imagenet-r":
        return generate_imagenet_r(num_classes=num_classes)
    root_path = f"{MyPath.db_root_dir('imagenet')}{version}"
    folders =folder_set[version]
    folders = folders.split()
    train_path = os.path.join(root_path,"ImgNetTrain")
    val_path = os.path.join(root_path,"ImgNetVal")
    IMG_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
    np.random.seed(0)
    # cls_num = 50
    os.makedirs(f'./data/imagenet/{version}', exist_ok=True)
    fout_train_label = open(f'./data/imagenet/{version}/{version}_label_{num_classes}_imbalance_{imbalance_factor}.txt', 'w')
    fout_val = open(f'./data/imagenet/{version}/{version}_numclass_{num_classes}_val.txt', 'w')

    assert num_classes<=len(folders)
    nums = torch.zeros(len(folders))
    # val_cls = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    for i, folder_name in enumerate(folders):
        train_files = []
        val_files = []
        for extension in IMG_EXTENSIONS:
            train_files.extend(glob.glob(os.path.join(train_path, folder_name, '*' + extension)))
            val_files.extend(glob.glob(os.path.join(val_path, folder_name, '*' + extension)))


        for filename in train_files:
            filepath = os.path.join(folder_name, filename.split("/")[-1])
            # if i < cls_num:
            if np.random.rand() <= (imbalance_factor ** (i / (num_classes - 1))):
                nums[i] = nums[i] + 1
                fout_train_label.write('%s %d\n' % (filepath, i))

        for filename in val_files:
            filepath = os.path.join(folder_name, filename.split("/")[-1])
            fout_val.write('%s %d\n' % (filepath, i))

    fout_train_label.close()
    # fout_train_unlabel.close()
    fout_val.close()
    # fout_train_label_val.close()
    # fout_train_unlabel_val.close()
    print("--Data has been generated!!!--")
    with open(f"./data/imagenet/{version}/{version}_numclass_{num_classes}_imbalance_{imbalance_factor}.pkl", "wb") as f:
        pkl.dump(nums, f)
    return nums

def generate_imagenet_r(root = MyPath.db_root_dir('imagenet-r'), test_num=20, num_classes=200):
    tmp = os.listdir(os.path.join(root))
    folders = [i for i in tmp if os.path.isdir(os.path.join(root, i))]
    version = "imagenet-r"
    IMG_EXTENSIONS = ['jpg', 'jpeg', 'JPG', 'JPEG']
    np.random.seed(0)
    # cls_num = 50
    os.makedirs(f'./data/imagenet/{version}', exist_ok=True)
    fout_train_label = open(f'./data/imagenet/{version}/{version}_label_{num_classes}.txt', 'w')
    fout_val = open(f'./data/imagenet/{version}/{version}_numclass_{num_classes}_val.txt', 'w')

    total_classes_num = len(folders)
    assert num_classes<=total_classes_num
    class_interval= total_classes_num // num_classes
    all_classes = [i*class_interval for i in range(num_classes)]
    nums = torch.zeros(num_classes)
    # val_cls = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
    all_train_files=[]
    all_val_files=[]
    nums_dict = {}
    for i in all_classes:
        folder_name = folders[i]

        # select n images as testset for each class
        all_files =[]
        for extension in IMG_EXTENSIONS:
            all_files.extend(glob.glob(os.path.join(root, folder_name, '*' + extension)))
        val_files = np.random.choice(all_files, 20, replace=False)
        train_files = list(set(all_files) - set(val_files))

        all_train_files.append(train_files)
        all_val_files.append(val_files)
        nums_dict[i] = len(train_files)

    # reorder by sort
    key_order=sorted(nums_dict, key=nums_dict.get, reverse=True)
    for i, key in enumerate(key_order):
        nums[i] = nums_dict[key]
        train_files = all_train_files[key]
        val_files = all_val_files[key]
        folder_name = folders[key]
        for filename in train_files:
            filepath = os.path.join(folder_name, filename.split("/")[-1])
            fout_train_label.write('%s %d\n' % (filepath, i))
        for filename in val_files:
            filepath = os.path.join(folder_name, filename.split("/")[-1])
            fout_val.write('%s %d\n' % (filepath, i))

    fout_train_label.close()
    fout_val.close()
    with open(f"./data/imagenet/{version}/{version}_key_order.pkl", "wb") as f:
        pkl.dump(key_order, f)

    print("--Data has been generated!!!--")
    with open(f"./data/imagenet/{version}/{version}_numclass_{num_classes}.pkl", "wb") as f:
        pkl.dump(nums, f)
    return nums

if __name__ == '__main__':
    # generate(100,10)
    # get_ImageNet_datasets(100, 0.1, version="imagenet100")
    # generate(50,50,50)
    nums = generate(200, 0.1, version="imagenet-r")
    from data import plot_distribution
    plot_distribution(nums, f"ImgNet-R",train=True)
