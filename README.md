# PPOT: Progressive Partial Optimal Transport for Deep Imbalanced Clustering


## Installation
```shell
git clone https://github.com/rhfeiyang/PPOT.git
cd PPOT
conda env create -f environment.yml
```


## Training

### Setup
Follow the steps below to setup the datasets:
- Change the file paths to the datasets in `utils/mypath.py`, e.g. `/path/to/cifar100`.


Our experimental evaluation includes the following datasets: CIFAR100, imagenet-r and iNaturalist18. Our code will build the imbalanced datasets automatically.

### Train model
For training on different datasets, args `--train_db_name` and `--val_db_name` should be specified. For example:
```shell
# For cifar100(imbalance ratio 100):
python train.py --train_db_name cifar_im --val_db_name cifar_im --imbalance_ratio 0.01 --num_classes 100 --num_heads 2  --output_dir experiment/PPOT/cifar100/ckpts
# For imagenet-r:
python train.py --train_db_name imagenet-r_im --val_db_name imagenet-r_im --num_classes 200 --num_heads 1 --output_dir experiment/PPOT/imagenet-r/ckpts
# For iNature100, 500, 1000:
python train.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 100 --num_heads 1 --output_dir experiment/PPOT/inature100/ckpts
python train.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 500 --num_heads 1 --output_dir experiment/PPOT/inature500/ckpts
python train.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 1000 --num_heads 1 --output_dir experiment/PPOT/inature1000/ckpts
```
### Remarks
We use multi-heads(2 heads of the same number of clusters) on CIFAR100, while one head for others. If you want to try overclustering, for example, heads of 100 and 200 clusters, you should set `--num_heads 2 --num_classes 100 200`. For overclustering, only the first head will be used for evaluation.



### Evaluation
For evaluation, just change the script file to "eval.py". Models in "output_dir" will be loaded. For example:
```shell
# For cifar100(imbalance ratio 100):
python eval.py --train_db_name cifar_im --val_db_name cifar_im --imbalance_ratio 0.01 --num_classes 100 --num_heads 2  --output_dir experiment/PPOT/cifar100/ckpts
# For imagenet-r:
python eval.py --train_db_name imagenet-r_im --val_db_name imagenet-r_im --num_classes 200 --num_heads 1 --output_dir experiment/PPOT/imagenet-r/ckpts
# For iNature100, 500, 1000:
python eval.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 100 --num_heads 1 --output_dir experiment/PPOT/inature100/ckpts
python eval.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 500 --num_heads 1 --output_dir experiment/PPOT/inature500/ckpts
python eval.py --train_db_name iNature_im --val_db_name iNature_im --num_classes 1000 --num_heads 1 --output_dir experiment/PPOT/inature1000/ckpts
```

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).