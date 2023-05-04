import random
import numpy as np
import os
import json
import argparse
import torch
import dataloaders
import models
import math
from utils import Logger
from trainer import Trainer
from utils.losses import abCE_loss, CE_loss, consistency_weight

import torch.multiprocessing as mp
import torch.distributed as dist

def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(gpu, ngpus_per_node, config, resume, test):
    
    if gpu == 0:
        train_logger = Logger()
    else:
        train_logger = None

    config['rank'] = gpu + ngpus_per_node * config['n_node']

    torch.cuda.set_device(gpu)
    assert config['train_supervised']['batch_size'] % config['n_gpu'] == 0
    assert config['train_unsupervised']['batch_size'] % config['n_gpu'] == 0
    assert config['val_loader']['batch_size'] % config['n_gpu'] == 0
    config['train_supervised']['batch_size'] = int(config['train_supervised']['batch_size'] / config['n_gpu'])
    config['train_unsupervised']['batch_size'] = int(config['train_unsupervised']['batch_size'] / config['n_gpu'])
    config['val_loader']['batch_size'] = int(config['val_loader']['batch_size'] / config['n_gpu'])
    config['train_supervised']['num_workers'] = int(config['train_supervised']['num_workers'] / config['n_gpu'])
    config['train_unsupervised']['num_workers'] = int(config['train_unsupervised']['num_workers'] / config['n_gpu'])
    config['val_loader']['num_workers'] = int(config['val_loader']['num_workers'] / config['n_gpu'])
    dist.init_process_group(backend='nccl', init_method=config['dist_url'], world_size=config['world_size'], rank=config['rank'])

    seed = config['random_seed']

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.

    # DATA LOADERS
    config['train_supervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['n_labeled_examples'] = config['n_labeled_examples']
    config['val_loader']['n_labeled_examples'] = config['n_labeled_examples']
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    config['train_supervised']['data_dir'] = config['data_dir']
    config['train_unsupervised']['data_dir'] = config['data_dir']
    config['val_loader']['data_dir'] = config['data_dir']
    config['train_supervised']['datalist'] = config['datalist']
    config['train_unsupervised']['datalist'] = config['datalist']
    config['val_loader']['datalist'] = config['datalist']
    config['train_supervised']['folder'] = 'train'
    config['train_unsupervised']['folder'] = 'train'
    config['val_loader']['folder'] = 'test'
    config['train_supervised']['config_dir'] = config['config_dir']
    config['train_unsupervised']['config_dir'] = config['config_dir']
    config['val_loader']['config_dir'] = config['config_dir']
    config['model']['epochs'] = config['trainer']['epochs']


    if config['dataset'] == 'bcss':
        sup_dataloader = dataloaders.BCSS
        unsup_dataloader = dataloaders.PairBCSS
    elif config['dataset'] == 'monuseg':
        sup_dataloader = dataloaders.MoNuSeg
        unsup_dataloader = dataloaders.PairMoNuSeg

    supervised_loader = sup_dataloader(config['train_supervised'])
    unsupervised_loader = unsup_dataloader(config['train_unsupervised'])
    val_loader = sup_dataloader(config['val_loader'])

    # set the target stain for unsupervised data
    # unsupervised_loader.dataset.set_target_stain_matrix(supervised_loader.dataset.get_stain_matrix())
    # supervised_loader.dataset.set_target_stain_matrix(unsupervised_loader.dataset.get_stain_matrix())
    # val_loader.dataset.set_target_stain_matrix(unsupervised_loader.dataset.get_stain_matrix())

    #### Fix iter_per_epoch ####
    iter_per_epoch = 100

    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                      rampup_ends=rampup_ends)

    sup_loss = CE_loss
    model = models.CAC(num_classes=val_loader.dataset.num_classes, conf=config['model'],
    						sup_loss=sup_loss, ignore_index=val_loader.dataset.ignore_index)

    # model = models.CCT(num_classes=val_loader.dataset.num_classes, conf=config['model'],
    # 						sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
    # 						weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'],
    #                         ignore_index=val_loader.dataset.ignore_index)

    # model = models.CDCL(num_classes=val_loader.dataset.num_classes, conf=config['model'],
    #                    sup_loss=sup_loss, ignore_index=val_loader.dataset.ignore_index)

    if gpu == 0:
        print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        train_logger=train_logger,
        gpu=gpu, 
        test=test)

    trainer.train()

def find_free_port():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='/home/saad/dev/cac_segmentation/configs/cac_monuseg_config_file.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-t', '--test', default=False, type=bool,
                        help='whether to test')
    args = parser.parse_args()

    config = json.load(open(args.config))
    torch.backends.cudnn.benchmark = True
    port = find_free_port()
    config['dist_url'] = f"tcp://127.0.0.1:{port}"
    print(config['dist_url'])
    config['n_node'] = 0 #only support 1 node
    config['world_size'] = config['n_gpu']
    mp.spawn(main, nprocs=config['n_gpu'], args=(config['n_gpu'], config, args.resume, args.test))


