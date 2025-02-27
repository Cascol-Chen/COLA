from logging import debug
import os
import time
import argparse
import json
import random
import math

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask, imagenet_a_mask

import torch    
import torch.nn.functional as F
import numpy as np

import tta_library.eata as eata
import tta_library.sar as sar
import tta_library.deyo as deyo
from torch.utils.tensorboard import SummaryWriter
import timm

from tta_library.cola import CoLAViT
from agents.multi_optimizer import CoLAOptimizer
from agents.fp_agents import FPAgent
import glob

def validate_adapt(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')
    
    with torch.no_grad():
        end = time.time()
        for i, dl in enumerate(val_loader):
            images, target = dl[0].cuda(), dl[1].cuda()
            output = model(images)
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            del output

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
                
            if i % 10 == 0:
                # logger.info(adapt_model.alpha.data)
                progress.display(i)

    return top1.avg, top5.avg, model

def get_args():

    parser = argparse.ArgumentParser(description='PyTorch ImageNet-C Testing')

    # path of data, output dir
    parser.add_argument('--data', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_v2', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_sketch', default='/dockerdata/imagenet', help='path to dataset')
    parser.add_argument('--data_corruption', default='/dockerdata/imagenet-c', help='path to corruption dataset')
    parser.add_argument('--data_rendition', default='/dockerdata/imagenet-r', help='path to corruption dataset')
    parser.add_argument('--data_adv', default='/dockerdata/imagenet-a', help='path to corruption dataset')
    parser.add_argument('--output', default='/apdcephfs/private_huberyniu/etta_exps/camera_ready_debugs', help='the output directory of this experiment')

    # general parameters, dataloader parameters
    parser.add_argument('--seed', default=2020, type=int, help='seed for initializing training. ')
    parser.add_argument('--gpu', default=0, type=int, help='GPU id to use.')
    parser.add_argument('--debug', default=False, type=bool, help='debug or not.')
    parser.add_argument('--workers', default=2, type=int, help='number of data loading workers (default: 4)')
    parser.add_argument('--batch_size', default=64, type=int, help='mini-batch size (default: 64)')
    parser.add_argument('--if_shuffle', default=True, type=bool, help='if shuffle the test set.')

    parser.add_argument('--fisher_clip_by_norm', type=float, default=10.0, help='Clip fisher before it is too large')

    # dataset settings
    parser.add_argument('--level', default=5, type=int, help='corruption level of test(val) set.')
    parser.add_argument('--corruption', default='gaussian_noise', type=str, help='corruption type of test(val) set.')
    parser.add_argument('--rotation', default=False, type=bool, help='if use the rotation ssl task for training (this is TTTs dataloader).')

    # model name, support resnets
    parser.add_argument('--arch', default='resnet50', type=str, help='the default model architecture')

    # eata settings
    parser.add_argument('--fisher_size', default=2000, type=int, help='number of samples to compute fisher information matrix.')
    parser.add_argument('--fisher_alpha', type=float, default=2000., help='the trade-off between entropy and regularization loss, in Eqn. (8)')
    parser.add_argument('--e_margin', type=float, default=math.log(1000)*0.40, help='entropy margin E_0 in Eqn. (3) for filtering reliable samples')
    parser.add_argument('--d_margin', type=float, default=0.05, help='\epsilon in Eqn. (5) for filtering redundant samples')
    

    # overall experimental settings
    parser.add_argument('--exp_type', default='continual', type=str, help='continual or each_shift_reset') 
    # 'cotinual' means the model parameters will never be reset, also called online adaptation; 
    # 'each_shift_reset' means after each type of distribution shift, e.g., ImageNet-C Gaussian Noise Level 5, the model parameters will be reset.
    parser.add_argument('--algorithm', default='eta', type=str, help='eata or eta or tent')
    parser.add_argument('--ensemble_weights', default=None, type=float, help='weight ensembling from ICML Anonymous')
    parser.add_argument('--ema_weights', default=None, type=float, help='ema weights for EMA')
    parser.add_argument('--tag', default='', type=str, help='the tag of experiment')
    parser.add_argument('--resume', default=None, type=str, help='pretrained weights')
    parser.add_argument('--sar_margin_e0', default=math.log(1000)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')

    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    # args.if_shuffle = False

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    algorithm_name = args.algorithm + args.tag
    args.output += '/' + algorithm_name + '/'

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)

    logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    logger.info('using model vitbase_timm')

    net = timm.create_model('vit_base_patch16_224', pretrained=True)

    ### 加载已经保存的权重 ####
    vectors_root = args.resume + '/*'
    weight_paths = glob.glob(vectors_root)

    net = net.cuda()
    common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog', 'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']   
    
    logger.info(args)
    logger.info(common_corruptions)

    if args.algorithm == 'cola-fp':
        net = CoLAViT(net, fp_agent_mode_on=True, fp_temperature=5, logger=logger)
        net = eata.configure_model(net)
        net.load_weights_from_files('./', weight_paths)
        adapt_model = net

    corrupt_acc = []
    for corrupt in common_corruptions:
        args.corruption = corrupt
        logger.info(args.corruption)

        if args.corruption == 'rendition':
            adapt_model.imagenet_mask = imagenet_r_mask
        elif args.corruption == 'adversial':
            adapt_model.imagenet_mask = imagenet_a_mask
        else:
            adapt_model.imagenet_mask = None

        val_dataset, val_loader = prepare_test_data(args)
        top1, top5, adapt_model = validate_adapt(val_loader, adapt_model, args)
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.5f} and Top-5 Accuracy: {top5:.5f}")
        corrupt_acc.append(top1)

    logger.info(f'mean acc of corruption: {sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0}')
    logger.info(f'corrupt acc list: {[_.item() for _ in corrupt_acc]}')
