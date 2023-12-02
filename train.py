"""
Authors: Hui Ren
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""
import argparse
import os
# import torch

from termcolor import colored
from utils.config import create_config
from utils.common_config import get_train_transformations, get_val_transformations,\
                                get_train_dataset, get_train_dataloader,\
                                get_val_dataset, get_val_dataloader,\
                                get_optimizer, get_model,\
                                adjust_learning_rate
from utils.evaluate_utils import get_predictions, hungarian_evaluate
from utils.train_utils import sk_train
from data.wrapper import two_view_wrapper

from losses.losses import SK_loss

from models import model_statistics
from models.dino import get_parameter_with_grad
import time
from utils.memory import LogitsMemory
import torch
# import torch.nn.functional as F
import numpy as np

parser = argparse.ArgumentParser(description='PPOT train')
parser.add_argument('--setup', default="cluster")
parser.add_argument('-c','--continue_train', default=False, action='store_true', help='Continue training from checkpoint')

parser.add_argument("--pretext_dir", default=None, type=str, help="pretext_dir")
parser.add_argument("--output_dir", default="experiments/", type=str, help="output_dir")
parser.add_argument("--gamma_bound", default=1.0, type=float)
parser.add_argument("--head_type", default="cos", type=str, help="head_type: linear/cos")

#dataset
parser.add_argument("--num_workers", default=8, type=int)
parser.add_argument("--train_db_name", default="cifar_im", type=str, help="cifar_im, iNature_im , imagenet-r_im")
parser.add_argument("--val_db_name", default="cifar_im", type=str, help="cifar_im, iNature_im , imagenet-r_im")
parser.add_argument('--imbalance_ratio', default=0.01, type=float, help='imbalance_ratio for dataset(cifar)')

parser.add_argument("--num_heads",default=2,type=int)
parser.add_argument("--num_classes",default=[100],type=int,nargs="+")
parser.add_argument("--epochs",default=50,type=int)
parser.add_argument("--train_eval_interval", default=10, type=int, help = "eval interval for trainset")
parser.add_argument("--backbone", default="dino_vitb16", type=str, help="backbone: dino_vitb16")
'''SK'''
parser.add_argument("--sk_type", default="ppot", type=str, help="uot, ppot, pot")
parser.add_argument("--sk_factor", default=0.1, type=float)
parser.add_argument("--sk_iter", default=3, type=int)
parser.add_argument("--sk_iter_limit", default=1000, type=int)
parser.add_argument("--batch_size", default=512,type=int)
parser.add_argument("--eval_batch_size", default=1024,type=int)

parser.add_argument("--sk_confusion",default=False,action="store_true",help="save sk confusion matrix")
parser.add_argument("--sk_w",default=1.0,type=float)
parser.add_argument("--rho_base",default=0.1,type=float)
parser.add_argument("--rho_upper",default=1.0,type=float)
parser.add_argument("--rho_fix", default=False, action='store_true', help='fix rho')
parser.add_argument("--rho_strategy", default="sigmoid", type=str, help="sigmoid/linear")

parser.add_argument("--bank_use",default=True ,action="store_true",help="use logits bank")
parser.add_argument("--bank_factor", default=10, type=int,help="factor for logits bank, factor*batch_size")
parser.add_argument("--model_select",default="loss",type=str,help="loss/last")
parser.add_argument("--select_set",default="train",type=str)

parser.add_argument("--detail", default=False, action='store_true', help='detail')
parser.add_argument("--label_quality_show", default=False, action='store_true', help='show pseudo label quality')
# optimizer
parser.add_argument("--optimizer", default="adam")
parser.add_argument("--lr", default=0.0005, type=float)
parser.add_argument("--weight_decay",default=0.0,type=float)
# scheduler
parser.add_argument("--scheduler", default="cosine", type=str)
parser.add_argument("--lr_decay_rate", default=0.1, type=float)

def eval(config,dataloader, model, head_select, confusion=False,class_names = None, confusion_file = None):
    predictions = get_predictions(config, dataloader, model)
    clustering_stats = hungarian_evaluate(head_select, predictions,class_names=class_names,num_classes=config["num_classes"][0],
                                          compute_confusion_matrix=confusion, confusion_matrix_file=confusion_file)
    return clustering_stats

def main():
    args = parser.parse_args()
    p = create_config(args=args)
    print(colored(p, 'red'))
    multi_head = args.num_heads>1 and len(args.num_classes)==1

    # CUDNN
    torch.backends.cudnn.benchmark = True

    # Model
    print(colored('Get model', 'blue'))
    model = get_model(p)
    # print(model)

    model = torch.nn.DataParallel(model)
    model = model.cuda()

    # Data
    print(colored('Get dataset and dataloaders', 'blue'))
    train_transformations = get_train_transformations(p)
    val_transformations = get_val_transformations(p)

    train_dataset = get_train_dataset(p, train_transformations, split='train')
    val_dataset = get_val_dataset(p, val_transformations)
    train_dataloader = get_train_dataloader(p, two_view_wrapper(train_dataset))

    train_dataset_for_eval = get_train_dataset(p, val_transformations, split='train')
    train_dataloader_for_eval = get_val_dataloader(p, train_dataset_for_eval)

    val_dataloader = get_val_dataloader(p, val_dataset)
    print('Train transforms:', train_transformations)
    print('Validation transforms:', val_transformations)
    print('Train samples %d - Val samples %d' %(len(train_dataset), len(val_dataset)))
    try:
        trainset_class= train_dataset.classes
        valset_class= val_dataset.classes
    except:
        trainset_class=None
        valset_class=None

    # Optimizer
    print(colored('Get optimizer', 'blue'))
    optimizer = get_optimizer(p, model)
    print(optimizer)

    criterion_sk = SK_loss(p,sk_type=p["sk_type"],factor=p["sk_factor"],num_iter=p["sk_iter"],total_iter=len(train_dataloader) * p['epochs'], start_iter=0)
    print(criterion_sk)

    # Checkpoint
    if args.continue_train and os.path.exists(p['cluster_checkpoint']):
        print(colored('Restart from checkpoint {}'.format(p['cluster_checkpoint']), 'blue'))
        checkpoint = torch.load(p['cluster_checkpoint'])
        model.load_state_dict(checkpoint['model'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        best_loss_head = checkpoint['best_loss_head']
        criterion_sk.i = start_epoch*len(train_dataloader)
        criterion_sk.logits_bank = checkpoint['logits_bank']

    else:
        print(colored('New train or No checkpoint file at {}'.format(p['cluster_checkpoint']), 'blue'))
        start_epoch = 0
        best_loss = 1e4
        best_loss_head = None

    # Loss function
    print(colored('Get loss', 'blue'))
    p["num_examples"] = len(train_dataset)
    p["log_upper_bounds"] = torch.log(torch.ones(p['num_classes'][0])/p['num_classes'][0])
    p["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    # Main loop
    print(colored('Starting main loop', 'blue'))
    if args.sk_confusion:
        os.makedirs(os.path.join(p['cluster_dir'],"sk_confusion"),exist_ok=True)
    if args.detail:
        os.makedirs(os.path.join(p['cluster_dir'],"labels"),exist_ok=True)
    model_statistics(model)
    for epoch in range(start_epoch, p['epochs']):
        print(colored('Epoch %d/%d' %(epoch+1, p['epochs']), 'yellow'))
        print(colored('-'*15, 'yellow'))
        # gamma = 1 - linear_rampup(epoch, p['epochs'])
        # criterion_sk.set_gamma(gamma)
        # print(f"Adjusted gamma to:{gamma}")
        # Adjust lr
        lr = adjust_learning_rate(p, optimizer, epoch)
        print('Adjusted learning rate to {:.5f}'.format(lr))
        if epoch == 1:
            heads = [args.num_classes[0]]*args.num_heads if multi_head else args.num_classes
            criterion_sk.logits_bank = [LogitsMemory(class_num, args.bank_factor*args.batch_size).cuda().init(train_dataloader,model,head) for head,class_num in enumerate(heads)] if args.bank_use else None

        # Train
        print('Train ...')
        current_time=time.time()
        train_stats= sk_train(train_dataloader, model, criterion_sk, optimizer, epoch,
                              sk_w=1e-6 if p["sk_type"] == "sla" and epoch == 0 else args.sk_w)
        print(f"train time:{time.time()-current_time}")

        # ----------------------------
        if args.select_set == "train":
            print(f"train_loss: {train_stats['total_loss']}")
            train_loss_factor = 1
            if "pot" in args.sk_type:
                train_loss_factor = 1/criterion_sk.sk[0].rho

            for head_id in range(len(train_stats['total_loss'])):
                train_stats['total_loss'][head_id] = train_stats['total_loss'][head_id]*train_loss_factor #+ head_label_kl[head_id]
            print(f"total_loss: {train_stats['total_loss']}")
            train_stats['lowest_loss_head'] = train_stats['total_loss'].argmin()
            train_stats['lowest_loss'] = train_stats['total_loss'][train_stats['lowest_loss_head']]

            if multi_head:
                lowest_loss_head = train_stats['lowest_loss_head']
                lowest_loss = train_stats['lowest_loss']
                print(f"lowest_loss_head:{lowest_loss_head}, lowest_loss:{lowest_loss}")
            else:
                lowest_loss_head=0
                lowest_loss=train_stats['total_loss'][0]

        else:
            raise NotImplementedError

        # --------------------------------
        ## check sk quality
        if args.label_quality_show:
            labels=torch.cat(criterion_sk.labels[lowest_loss_head],dim=0)
            label_distribute=torch.sum(labels,dim=0).cpu()
            print(f"SK label distribute for head {lowest_loss_head}:{np.sort(label_distribute.numpy())[::-1]}, normalized:{np.sort((label_distribute/label_distribute.sum()).numpy())[::-1] if p['num_classes'][0] <=10 else 'omit'}")

            sk_prediction, sk_prediction_top_rho = criterion_sk.prediction_log(top_rho=True)
            if len(args.num_classes)>1:
                sk_prediction=[sk_prediction[0]]
                sk_prediction_top_rho=[sk_prediction_top_rho[0]]
            quality = [hungarian_evaluate(head_id, sk_prediction,num_classes=p["num_classes"][0], compute_confusion_matrix=args.sk_confusion
                                          ,class_names=trainset_class,
                                           confusion_matrix_file=os.path.join(p['cluster_dir'],"sk_confusion", f'sk_confusion_matrix{epoch}_{head_id}.png')) for head_id in range(len(sk_prediction))]
            quality_top_rho=[hungarian_evaluate(head_id, sk_prediction_top_rho, num_classes=p["num_classes"][0], compute_confusion_matrix=args.sk_confusion
                                                , class_names=trainset_class,
                                                confusion_matrix_file=os.path.join(p['cluster_dir'],"sk_confusion", f'sk_confusion_matrix_top10_{epoch}_{head_id}.png')) for head_id in range(len(sk_prediction_top_rho))]
            for head_id in range(len(quality)):
                print(colored(f"Top_rho confidence for head {head_id}:{quality_top_rho[head_id]}", "blue"))
                print(colored(f"Sinhorn label quality for head {head_id}:{quality[head_id]}","blue"))
        ###
        if args.sk_type in ["ppot","pot"] :
            print(colored(f"rho:{criterion_sk.sk[0].rho}","blue"))

        if args.detail:
            torch.save({"rho": criterion_sk.sk[0].rho, "sk_prediction": sk_prediction, "lowest_loss_head":lowest_loss_head,
                        }, os.path.join(p['cluster_dir'], "labels", f"sk_detail_{epoch}_{round(criterion_sk.sk[0].rho,4)}.pth"))

        criterion_sk.reset()
        ##

        # update selected model
        if args.model_select == "loss":
            if lowest_loss < best_loss:
                print('New lowest loss: %.4f -> %.4f' %(best_loss, lowest_loss))
                print('Lowest loss head is %d' %(lowest_loss_head))
                best_loss = lowest_loss
                best_loss_head = lowest_loss_head
                torch.save({'model': get_parameter_with_grad(model.module), 'head': best_loss_head}, p['cluster_model'])

            else:
                print('No new lowest loss: %.4f -> %.4f' %(best_loss, lowest_loss))
                print('Lowest loss head is %d' %(best_loss_head))
        elif args.model_select == "last":
            best_loss_head=lowest_loss_head
        else:
            raise NotImplementedError

        # Evaluate
        print('Make prediction ...')

        if epoch % args.train_eval_interval == 0:
            clustering_stats = eval(p,train_dataloader_for_eval, model, lowest_loss_head, confusion=False)
            print(f"trainset result: {clustering_stats}")

        clustering_stats = eval(p,val_dataloader, model, lowest_loss_head, confusion=False)
        print(f"testset result: {clustering_stats}")

        # Checkpoint
        print('Checkpoint ...')
        torch.save({'optimizer': optimizer.state_dict(), 'model': get_parameter_with_grad(model),
                    'epoch': epoch + 1, 'best_loss': best_loss, 'best_loss_head': best_loss_head, 'logits_bank': criterion_sk.logits_bank},
                     p['cluster_checkpoint'])

    # Evaluate and save the final model
    if args.model_select == "loss":
        print(colored('Evaluate selected model at the end', 'blue'))
        model_checkpoint = torch.load(p['cluster_model'], map_location='cpu')
        model.module.load_state_dict(model_checkpoint['model'], strict=False)
        model_checkpoint_head = model_checkpoint['head']
    else:
        checkpoint = torch.load(p['cluster_checkpoint'], map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        model_checkpoint_head = checkpoint['best_loss_head']

    clustering_stats = eval(p, train_dataloader_for_eval, model, model_checkpoint_head, confusion=True,class_names=trainset_class,confusion_file=os.path.join(p['cluster_dir'], 'train_confusion_matrix.png'))
    print(f"trainset result: {clustering_stats}")

    if args.model_select == "loss":
        clustering_stats = eval(p, val_dataloader, model, model_checkpoint_head, confusion=True,class_names=valset_class,confusion_file=os.path.join(p['cluster_dir'], 'test_confusion_matrix.png'))
        print(f"testset result: {clustering_stats}")
    else:
        # save the last model
        torch.save({'model': get_parameter_with_grad(model.module), 'head': best_loss_head}, p['cluster_model'])

if __name__ == "__main__":
    main()
