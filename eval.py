
import argparse
import os

import torch
import yaml
from termcolor import colored
from utils.common_config import get_train_transformations, get_val_transformations, \
    get_train_dataset, get_train_dataloader, \
    get_val_dataset, get_val_dataloader, \
    get_optimizer, get_model
from utils.evaluate_utils import get_predictions, hungarian_evaluate

from PIL import Image
import re

parser = argparse.ArgumentParser(description='Evaluate models')
# parser.add_argument('--model', help='Location where model is saved')
parser.add_argument("--output_dir", default="tmp/", type=str, help="output_dir")
parser.add_argument('--visualize_prototypes', action='store_true', 
                    help='Show the prototpye for each cluster')
parser.add_argument("--train_db_name", default="cifar_im", type=str, help="cifar_im, iNature_im , imagenet-r_im")
parser.add_argument("--val_db_name", default="cifar_im", type=str, help="cifar_im, iNature_im , imagenet-r_im")
parser.add_argument('--imbalance_ratio', default=0.01, type=float, help='imbalance_ratio for dataset')
parser.add_argument("--num_classes",default=[100],type=int,nargs="+")
parser.add_argument("--backbone", default="dino_vitb16", type=str, help="backbone: resnet18/resnet50/dino_vitb16")

parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--model_take", default="select", help="ckpt/select")
parser.add_argument("--no_train", default=False, action="store_true")
parser.add_argument("--no_test", default=False, action="store_true")
parser.add_argument("--no_cluster" ,default=False, action="store_true")
parser.add_argument("--no_selflabel", default=False, action="store_true")
args = parser.parse_known_args()[0]

# def target_num_count(p):
#     p['train_db_name'] = p['val_db_name']
#     train_transformations = get_train_transformations(p)
#     train_dataset = get_train_dataset(p, train_transformations,
#                                       split='train', to_neighbors_dataset = False,indices=None)
#     train_dataloader = get_val_dataloader(p, train_dataset)
#     nums = torch.zeros(p['num_classes'][0])
#     for batch in train_dataloader:
#         target = batch['target']
#         for i in range(p['num_classes'][0]):
#             nums[i] += torch.sum(target==i)
#     print(f"train target num count:{nums}")
#     return nums

def eval(dataset, model, head_select, config, setname, save_path=None):
    dataloader = get_val_dataloader(config, dataset)
    if save_path is not None and os.path.exists(save_path):
        predictions = torch.load(save_path)
    else:
        predictions = get_predictions(config, dataloader, model)
        if save_path is not None:
            torch.save(predictions,save_path)
    # print(f"label distribution: {predictions[0]['targets'].bincount()}")
    select_result=None
    for i in range(len(predictions)):
        clustering_stats = hungarian_evaluate(i, predictions,
                                              compute_confusion_matrix=False)
        print(f"{setname} head {i} result: {clustering_stats}")
        if i == head_select:
            select_result = clustering_stats

    print(f"{setname} result: {select_result}")


def main():
    # Read config file
    config={}
    config['batch_size'] = 512
    config.update(args.__dict__)
    config['setup'] = "cluster"
    print(config)

    # Get dataset
    print(colored('Get validation dataset ...', 'blue'))
    transforms = get_val_transformations(config)
    dataset_test = get_val_dataset(config, transforms)
    dataset_train = get_train_dataset(config, transforms,split="train")


    if args.model_take == "ckpt":
        cluster_model_path = os.path.join(args.output_dir, "cluster/checkpoint.pth.tar")
    elif args.model_take == "select":
        cluster_model_path = os.path.join(args.output_dir, "cluster/model.pth.tar")
    else:
        raise NotImplementedError
    selflabel_model_path = os.path.join(args.output_dir, "selflabel/checkpoint.pth.tar")
    path = []
    if not args.no_cluster and os.path.exists(cluster_model_path):
        print(f"cluster model path:{cluster_model_path}")
        path.append(("cluster", cluster_model_path))
    if not args.no_selflabel and os.path.exists(selflabel_model_path):
        print(f"selflabel model path:{selflabel_model_path}")
        path.append(("selflabel", selflabel_model_path))

    class_num_pattern = r'head.(\d+).'
    for name, model_path in path:
        print(f"========{name}========")
        state_dict = torch.load(model_path, map_location='cpu')
        head_num = 1
        head_select=0
        if "model" in state_dict:
            model_state = state_dict['model']
            if 'best_loss_head' in state_dict:
                head_select = state_dict['best_loss_head']
            elif "head" in state_dict:
                head_select = state_dict['head']

        else:
            model_state = state_dict
        head_type="linear"
        num_classes = []
        for key in model_state.keys():
            if "head" in key:
                tmp = re.findall(class_num_pattern, key)
                if tmp is not None and "weight" in key:
                    head_num = max(head_num, int(tmp[0])+1)
                    num_classes.append(model_state[key].shape[0])

            if "embedding.weight_v" in key:
                head_type="cos"
        config['num_classes'] = num_classes
        config['head_type'] = head_type
        config['num_heads'] = head_num
        print(f"head type for {name}:{head_type}")
        print(f"head num for {name}:{head_num}")
        print(f"num classes for {name}:{num_classes}")
        model = get_model(config, model_path)
        model = torch.nn.DataParallel(model)

        if "module" in list(model_state.keys())[0]:
            missing = model.load_state_dict(model_state, strict=False)
        else:
            missing = model.module.load_state_dict(model_state, strict=False)
        print(missing)
        model.cuda()
        if not args.no_train:
            eval(dataset_train, model, head_select, config, "trainset",save_path=os.path.join(os.path.dirname(model_path),"output_train.pth"))
        if not args.no_test:
            eval(dataset_test, model, head_select, config, "test",save_path=os.path.join(os.path.dirname(model_path),"output_test.pth"))


if __name__ == "__main__":
    print("EVAL_IM")
    main()
    print("Complete.")
