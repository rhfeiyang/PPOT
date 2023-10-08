"""
Authors: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import os
import yaml
from easydict import EasyDict
from utils.utils import mkdir_if_missing

def create_config(args):
    # Config for environment path

    cfg = EasyDict()
    cfg.update(args.__dict__)
    output_dir = args.output_dir
    mkdir_if_missing(output_dir)

    cfg['train_db_name']=args.train_db_name
    if cfg['setup'] in ['cluster']:
        if "cluster_dir" in args.__dict__ and args.cluster_dir is not None:
            cluster_dir = args.cluster_dir
        else:
            cluster_dir = os.path.join(output_dir, 'cluster')
        mkdir_if_missing(cluster_dir)
        cfg['cluster_dir'] = cluster_dir
        cfg['cluster_checkpoint'] = os.path.join(cluster_dir, 'checkpoint.pth.tar')
        cfg['cluster_model'] = os.path.join(cluster_dir, 'model.pth.tar')

    return cfg
