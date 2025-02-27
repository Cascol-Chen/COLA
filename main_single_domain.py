import os
import time
import argparse
import random
import math
import copy

from utils.utils import get_logger
from utils.cli_utils import *
from dataset.selectedRotateImageFolder import prepare_test_data
from dataset.ImageNetMask import imagenet_r_mask, imagenet_a_mask

import torch    
import torch.nn.functional as F
import numpy as np

import tta_library.tent as tent
import tta_library.eata as eata
import tta_library.sar as sar
import tta_library.deyo as deyo
from tta_library.sam import SAM
from tta_library.t3a import T3A
from tta_library.lame import LAME
import tta_library.cotta as cotta

import tta_library.f_sam as f_sam

import timm

from tta_library.cola import CoLAViT, CoLAOptimizer

def get_agent(args, net, weight_paths):
    net = copy.deepcopy(net)
    if args.algorithm == 'eta':
        net = eata.configure_model(net)
        params, param_names = eata.collect_params(net)
        optimizer = torch.optim.SGD(params, 0.001, momentum=0.9)
        adapt_model = eata.EATA(net, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'sar':
        net = sar.configure_model(net)
        params, _ = sar.collect_params(net)
        base_optimizer = torch.optim.SGD
        optimizer = SAM(params, base_optimizer, lr=0.001, momentum=0.9)
        # NOTE: set margin_e0 to 0.4*math.log(200) on ImageNet-R
        adapt_model = sar.SAR(net, optimizer)
    elif args.algorithm == 'deyo':
        net = deyo.configure_model(net)
        params, param_names = deyo.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        adapt_model = deyo.DeYO(net, optimizer)
    elif args.algorithm == 'cotta':
        net = cotta.configure_model(net)
        params, _ = cotta.collect_params(net)
        optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9)
        adapt_model = cotta.CoTTA(net, optimizer, steps=1, episodic=False)
    elif args.algorithm == 'eata':
        # compute fisher informatrix
        args.corruption = 'original'
        fisher_dataset, fisher_loader = prepare_test_data(args)
        fisher_dataset.set_dataset_size(args.fisher_size)
        fisher_dataset.switch_mode(True, False)

        net = eata.configure_model(net)
        params, param_names = eata.collect_params(net)
        ewc_optimizer = torch.optim.SGD(params, 0.001)
        fishers = {}
        train_loss_fn = nn.CrossEntropyLoss().cuda()
        for iter_, (images, targets) in enumerate(fisher_loader, start=1):      
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                targets = targets.cuda(args.gpu, non_blocking=True)
            outputs = net(images)
            _, targets = outputs.max(1)
            loss = train_loss_fn(outputs, targets)
            loss.backward()
            for name, param in net.named_parameters():
                if param.grad is not None:
                    if iter_ > 1:
                        fisher = param.grad.data.clone().detach() ** 2 + fishers[name][0]
                    else:
                        fisher = param.grad.data.clone().detach() ** 2
                    if iter_ == len(fisher_loader):
                        fisher = fisher / iter_
                    fishers.update({name: [fisher, param.data.clone().detach()]})
            ewc_optimizer.zero_grad()
        logger.info("compute fisher matrices finished")
        del ewc_optimizer

        optimizer = torch.optim.SGD(params, 0.001, momentum=0.9)
        adapt_model = eata.EATA(net, optimizer, fishers, args.fisher_alpha, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'eta-cola':
        net = CoLAViT(net, 0.01, logger=logger, domain_detect_on=False, save_per_iteration=10, auto_remove_on=True, max_num_vectors=32)
        net = eata.configure_model(net)
        net.load_weights_from_files('./', weight_paths)
        params = net.collect_params()

        alpha_optimizer = torch.optim.AdamW([
            {'params': params['alpha'], 'lr': 0.1, 'weight_decay': 0.1},
            {'params': params['alpha_scale'], 'lr': 0.1}
        ])
        eps_optimizer = torch.optim.SGD(params['epsilon_weight'] + params['epsilon_bias'], 0.001, momentum=0.9, weight_decay=0.)
        optimizer = CoLAOptimizer(len(weight_paths), alpha_optimizer, eps_optimizer)
        net.cola_optimizer = optimizer
        adapt_model = eata.EATA(net, optimizer, e_margin=args.e_margin, d_margin=args.d_margin)
    elif args.algorithm == 'deyo-cola':
        net = CoLAViT(net, 0.01, logger=logger, domain_detect_on=False, save_per_iteration=10, auto_remove_on=True, max_num_vectors=32)
        net = deyo.configure_model(net)
        net.load_weights_from_files('./', weight_paths)
        params = net.collect_params()

        alpha_optimizer = torch.optim.AdamW([
            {'params': params['alpha'], 'lr': 0.1, 'weight_decay': 0.1},
            {'params': params['alpha_scale'], 'lr': 0.1}
        ])
        eps_optimizer = torch.optim.SGD(params['epsilon_weight'] + params['epsilon_bias'], 0.001, momentum=0.9, weight_decay=0.)
        optimizer = CoLAOptimizer(len(weight_paths), alpha_optimizer, eps_optimizer)
        net.cola_optimizer = optimizer
        adapt_model = deyo.DeYO(net, optimizer)
    elif args.algorithm == 'sar-cola':
        net = CoLAViT(net, 0.01, logger=logger, domain_detect_on=False, save_per_iteration=10, auto_remove_on=True, max_num_vectors=32)
        net = sar.configure_model(net)
        net.load_weights_from_files('./', weight_paths)
        params = net.collect_params()

        alpha_optimizer = torch.optim.AdamW([
            {'params': params['alpha'], 'lr': 0.1, 'weight_decay': 0.1},
            {'params': params['alpha_scale'], 'lr': 0.1}
        ])
        base_optimizer = torch.optim.SGD
        optimizer = f_sam.SAM(params['epsilon_weight'] + params['epsilon_bias'], base_optimizer, alpha_optimizer, len(weight_paths), lr=0.001, momentum=0.9, weight_decay=0.)
        net.cola_optimizer = optimizer
        adapt_model = sar.SAR(net, optimizer)
    return adapt_model


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
    parser.add_argument('--tag', default='', type=str, help='the tag of experiment')
    parser.add_argument('--resume', default=None, type=str, help='pretrained weights')
    parser.add_argument('--sar_margin_e0', default=math.log(1000)*0.40, type=float, help='the threshold for reliable minimization in SAR, Eqn. (2)')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    # we use random.shuffle to shuffle the ImageNet-C dataset, ensuring that eta and eta+cola
    # receive the same sequence of samples even cola uses some random numbers from torch for initialization
    args.if_shuffle = False

    # set random seeds
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
    
    algorithm_name = args.algorithm + args.tag
    weight_pool_output = args.output + '/weight_pool4/'
    args.output += '/' + algorithm_name + '/'
    model_output = args.output + '/model/'

    if not os.path.exists(args.output):
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(model_output, exist_ok=True)

    logger = get_logger(name="project", output_directory=args.output, log_name=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())+"-log.txt", debug=False)
    logger.info('using model vitbase_timm')
    net = timm.create_model('vit_base_patch16_224', pretrained=True)

    if args.resume is None or not os.path.isfile(args.resume):
        logger.info('converting timm weights to pth...')
        os.makedirs('weights', exist_ok=True)
        torch.save({'model': net.state_dict()},  'weights/original.pth') # convert to pth to ease implementation ^_^
        args.resume = 'weights/original.pth'

    # load saved model or domain vectors
    weight_paths = [args.resume]

    net = net.cuda()

    common_corruptions = ['gaussian_noise', 'defocus_blur', 'snow', 'contrast', 'shot_noise', 'glass_blur', 'frost', 'elastic_transform',  'impulse_noise',  'motion_blur', 'fog', 'pixelate', 'brightness', 'zoom_blur', 'jpeg_compression']

    if args.exp_type == 'label_shifts':
        indices_path = 'dataset/total_100000_ir_500000_class_order_shuffle_yes.npy'
        logger.info(f"label_shifts_indices_path is {indices_path}")
        dataset_indices = np.load(indices_path)
    elif args.exp_type == 'mix_shifts':
        datasets = []
        for cpt in common_corruptions:
            args.corruption = cpt
            logger.info(args.corruption)

            val_dataset, _ = prepare_test_data(args)
            datasets.append(val_dataset)

        from torch.utils.data import ConcatDataset
        mixed_dataset = ConcatDataset(datasets)
        logger.info(f"length of mixed dataset us {len(mixed_dataset)}")
        val_loader = torch.utils.data.DataLoader(mixed_dataset, batch_size=args.batch_size, shuffle=args.if_shuffle, num_workers=args.workers, pin_memory=True)
        common_corruptions = ['mix_shifts']
    elif 'each_shift_reset':
        pass
    else:
        assert False, NotImplementedError

    logger.info(args)
    logger.info(common_corruptions)

    corrupt_acc = []
    for corrupt in common_corruptions:
        # Cola has a different reset logit which won't discard the learned vectors
        # Thus, the simplest way for complete reset is to re-initialize an adapt model
        adapt_model = get_agent(args, net, weight_paths)
        args.corruption = corrupt
        logger.info(args.corruption)

        if args.corruption == 'rendition':
            adapt_model.imagenet_mask = imagenet_r_mask
        elif args.corruption == 'adversial':
            adapt_model.imagenet_mask = imagenet_a_mask
        else:
            adapt_model.imagenet_mask = None

        if args.corruption != 'mix_shifts':
            val_dataset, val_loader = prepare_test_data(args)
        if args.exp_type == 'label_shifts':
            val_dataset.set_specific_subset(dataset_indices.astype(int).tolist())
        
        top1, top5, adapt_model = validate_adapt(val_loader, adapt_model, args)
        logger.info(f"Under shift type {args.corruption} After {args.algorithm} Top-1 Accuracy: {top1:.5f} and Top-5 Accuracy: {top5:.5f}")
        corrupt_acc.append(top1)

    logger.info(f'mean acc of corruption: {sum(corrupt_acc)/len(corrupt_acc) if len(corrupt_acc) else 0}')
    logger.info(f'corrupt acc list: {[_.item() for _ in corrupt_acc]}')