import os
import cv2
import time
import random
import pickle
import numpy as np
from PIL import Image
from distutils.version import LooseVersion

from sacred import Experiment
from sacred.observers import FileStorageObserver
from easydict import EasyDict as edict

import torch
from torch.utils import data
import torch.nn.functional as F
from torchvision import transforms

from models.baseline_same import Baseline as UNet
from utils.loss import surface_normal_loss
from utils.misc import AverageMeter, get_optimizer
from utils.metric import eval_iou, eval_plane_prediction
from utils.disp import tensor_to_image
from utils.disp import colors_256 as colors

from scannet.dataloader_batch import ScannetDataset, Resizer, ToTensor
from logger import *

from depthnet.depthNet_model import depthNet, DepthRefineNet
from depthnet.losses import IdepthLoss, IdepthLoss_234, IdepthwithProbLoss, ProbLoss
from depthnet.depth_util import np2Depth, colorize_probmap, Depth2normal, normal2color, depth2color, \
    get_normal_by_planes

from fusion_depth.fuse_depth import get_warped_depth_loss

ex = Experiment()
ex.observers.append(FileStorageObserver.create('../experiments'))


def load_dataset(subset, cfg):
    transform = transforms.Compose([
        Resizer(image_height_expected=cfg.image_height,
                image_width_expected=cfg.image_width,
                depth_height_expected=cfg.image_height,
                depth_width_expected=cfg.image_width),
        ToTensor()
    ])

    is_shuffle = subset == 'train'

    loaders = data.DataLoader(
        ScannetDataset(list_filepath=cfg.list_filepath, transform=transform, root_dir=cfg.root_dir),
        batch_size=cfg.batch_size, shuffle=is_shuffle, num_workers=cfg.num_workers
    )

    return loaders


@ex.command
def train(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join('../experiments', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logger_dir = os.path.join('../experiments', str(_run._id), 'log')
    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)

    # logger
    logger = Logger(logger_dir)

    # build normal_network and depth_network
    depth_network = depthNet(idepth_scale=cfg.idepth_scale)
    # for p in depth_network.parameters():
    #     p.requires_grad = False

    depth_refine_network = DepthRefineNet(idepth_scale=cfg.idepth_scale)
    # set up optimizers
    network_params = list(depth_refine_network.parameters()) + list(depth_network.parameters())

    optimizer = get_optimizer(network_params, cfg.solver)


    if not cfg.resume_dir == 'None':
        print('resume training')
        checkpoint = torch.load(cfg.resume_dir)

        depth_network.load_state_dict(checkpoint['depth_network_state_dict'])

        try:
            depth_refine_network.load_state_dict(checkpoint['depth_refine_network_state_dict'])
        except:
            print("no checkpoint for refineNet")

        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
    else:
        global_step = 0
        start_epoch = 0

    # load nets into gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        gpu_num = torch.cuda.device_count()
        depth_network = torch.nn.DataParallel(depth_network)
        depth_refine_network = torch.nn.DataParallel(depth_refine_network)
    else:
        gpu_num = 1

    device_normal = torch.device('cuda:' + str(gpu_num - 1))
    device_net = torch.device(device)
    depth_network.to(device_net)
    depth_refine_network.to(device_net)

    depth2normal = Depth2normal(cfg.k_size)
    depth2normal.to(device_normal)

    # data loader
    data_loader = load_dataset('train', cfg.dataset)

    depth_network.train()
    depth_refine_network.train()

    criterion_234 = IdepthLoss_234()
    criterion_1 = IdepthLoss()
    criterion_idepth_prob = IdepthwithProbLoss()
    criterion_prob = ProbLoss()

    # main loop
    for epoch in range(start_epoch + 1, cfg.num_epochs):
        batch_time = AverageMeter()

        tic = time.time()
        for iter, sample in enumerate(data_loader):
            # try:
            image = sample['rgbs'].to(device)
            batch_size, views, c, h, w = image.shape
            instance = sample['plane_instance_segs']
            instance = instance.to(device)
            # semantic = sample['semantic'].to(device)
            gt_depth = sample['depths'].to(device)

            gt_seg = sample['plane_segs'].to(device)  # [b, views, h, w]
            normals_from_plane_para = sample['normals_from_plane_para'].to(device)

            gt_normal = sample['normals'].to(device)
            gt_normal_valid = gt_depth > 0.1
            plane_nums = sample['plane_nums']  # [b, views]
            # valid_region = sample['valid_region'].to(device)
            # gt_plane_instance_parameter = sample['plane_instance_parameter'].to(device)

            gt_cam = sample['cameras'].to(device)
            gt_disparity = sample['disparities'].to(device)

            idepth_preds_01, iconv_01 = depth_network(image[:, 0, :, :, :], image[:, 1, :, :, :],
                                                      gt_cam[:, 0, :, :, :], gt_cam[:, 1, :, :, :])
            idepth_preds_02, iconv_02 = depth_network(image[:, 0, :, :, :], image[:, 2, :, :, :],
                                                      gt_cam[:, 0, :, :, :], gt_cam[:, 2, :, :, :])

            ######################
            # left-right refine idepth
            ######################
            idepth_refined, prob_map = depth_refine_network(idepth01=idepth_preds_01[0],
                                                            idepth02=idepth_preds_02[0],
                                                            iconv01=iconv_01,
                                                            iconv02=iconv_02)

            loss_idepth_1 = (criterion_1(idepth_preds_01[0], gt_disparity[:, 0, :, :, :]) +
                             criterion_1(idepth_preds_02[0], gt_disparity[:, 0, :, :, :])) * 0.5

            loss_idepth_refined = criterion_1(idepth_refined, gt_disparity[:, 0, :, :, :])

            loss_idepth_234 = (criterion_234(idepth_preds_01, gt_disparity[:, 0, :, :, :]) +
                               criterion_234(idepth_preds_02, gt_disparity[:, 0, :, :, :])) * 0.5

            depth_preds_01 = torch.div(1.0, idepth_preds_01[0].squeeze(1))
            depth_preds_02 = torch.div(1.0, idepth_preds_02[0].squeeze(1))

            depth_refined = torch.div(1.0, idepth_refined.squeeze(1) + 1e-5)

            ####################################
            # prob loss
            #####################################
            prob_loss_depth = criterion_idepth_prob(idepth_refined, gt_disparity[:, 0, :, :, :], prob_map) + \
                              criterion_idepth_prob(depth_refined.unsqueeze(1), gt_depth[:, 0, :, :, :], prob_map)
            prob_loss_minusmean = 1 - prob_map.mean()

            prob_map_loss, prob_map_gt = criterion_prob(prob_map, idepth_refined, gt_disparity[:, 0, :, :, :])

            prob_loss = 5 * prob_loss_depth + prob_loss_minusmean  # + prob_map_loss

            intrinsic = gt_cam[:, 0, 1, 0:3, 0:3]
            intrinsic_inv = torch.inverse(intrinsic)

            normal_from_depth_01, _ = depth2normal(depth_preds_01.to(device_normal), intrinsic_inv.to(device_normal))
            normal_from_depth_02, _ = depth2normal(depth_preds_02.to(device_normal), intrinsic_inv.to(device_normal))
            normal_from_depth_refined, _ = depth2normal(depth_refined.to(device_normal),
                                                        intrinsic_inv.to(device_normal))

            normal_from_depth_01 = normal_from_depth_01.to('cuda:0')
            normal_from_depth_02 = normal_from_depth_02.to('cuda:0')
            normal_from_depth_refined = normal_from_depth_refined.to('cuda:0')

            normal_std = 0

            normal_by_planes = get_normal_by_planes(gt_normal[:, 0, :, :, :], instance[:, 0, :, :, :], plane_nums[:, 0])

            loss_depth_1 = (criterion_1(depth_preds_01.unsqueeze(1), gt_depth[:, 0, :, :, :]) +
                            criterion_1(depth_preds_02.unsqueeze(1), gt_depth[:, 0, :, :, :])) * 0.5

            loss_depth_refined = criterion_1(depth_refined.unsqueeze(1), gt_depth[:, 0, :, :, :])

            # calculate loss
            loss, loss_depth, loss_normal = 0., 0., 0.
            loss_normal_depth = 0
            loss_normal_depth_refined = 0
            for i in range(batch_size):
                if not cfg.use_normal_refined_by_planes:
                    _loss_normal_depth_01, mean_angle_depth_01 = surface_normal_loss(normal_from_depth_01[i:i + 1],
                                                                                     gt_normal[i:i + 1, 0, :, :, :],
                                                                                     gt_normal_valid[i:i + 1, 0, :, :,
                                                                                     :])
                    _loss_normal_depth_02, mean_angle_depth_02 = surface_normal_loss(normal_from_depth_02[i:i + 1],
                                                                                     gt_normal[i:i + 1, 0, :, :, :],
                                                                                     gt_normal_valid[i:i + 1, 0, :, :,
                                                                                     :])
                    _loss_normal_depth_refined, mean_angle_depth_refined = surface_normal_loss(
                        normal_from_depth_refined[i:i + 1],
                        gt_normal[i:i + 1, 0, :, :, :],
                        gt_normal_valid[i:i + 1, 0, :, :, :])
                    _loss_normal_depth = (_loss_normal_depth_01 + _loss_normal_depth_02) * 0.5

                    mean_angle_depth = (mean_angle_depth_01 + mean_angle_depth_02 + mean_angle_depth_refined) / 3.0
                else:
                    _loss_normal_depth_01, mean_angle_depth_01 = surface_normal_loss(normal_from_depth_01[i:i + 1],
                                                                                     normal_by_planes[i:i + 1, :, :, :],
                                                                                     gt_normal_valid[i:i + 1, 0, :, :,
                                                                                     :])
                    _loss_normal_depth_02, mean_angle_depth_02 = surface_normal_loss(normal_from_depth_02[i:i + 1],
                                                                                     normal_by_planes[i:i + 1, :, :, :],
                                                                                     gt_normal_valid[i:i + 1, 0, :, :,
                                                                                     :])
                    _loss_normal_depth_refined, mean_angle_depth_refined = surface_normal_loss(
                        normal_from_depth_refined[i:i + 1],
                        normal_by_planes[i:i + 1, :, :, :],
                        gt_normal_valid[i:i + 1, 0, :, :,
                        :])
                    _loss_normal_depth = (_loss_normal_depth_01 + _loss_normal_depth_02) * 0.5
                    mean_angle_depth = (mean_angle_depth_01 + mean_angle_depth_02 + mean_angle_depth_refined) / 3.0

                # planar segmentation iou

                loss_normal_depth += _loss_normal_depth
                loss_normal_depth_refined += _loss_normal_depth_refined
                # loss_pw += _pw_loss

            loss_depth /= batch_size
            loss_normal /= batch_size
            loss_normal_depth /= batch_size
            loss_normal_depth_refined /= batch_size
            normal_std /= batch_size
            # loss_pw /= batch_size

            loss += loss_idepth_1
            loss += loss_idepth_234
            if not ((~torch.isnan(loss_normal_depth)) & (~torch.isnan(loss_normal_depth_refined))):
                print('loss depth is nan')
                print(sample['filenames'])

                loss_train = loss_idepth_1 + loss_depth_1 + loss_depth_refined + loss_idepth_refined
            else:
                loss_train = loss_idepth_1 + loss_normal_depth + loss_depth_1 + loss_depth_refined + loss_idepth_refined + loss_normal_depth_refined
                loss_train += prob_loss

            ref_extrinsic = gt_cam[:, 0, 0, :, :]
            source1_extrinsic = gt_cam[:, 1, 0, :, :]
            pose1 = (source1_extrinsic @ torch.inverse(ref_extrinsic))[:, :3, :]
            warped_depth_loss_1 = get_warped_depth_loss(depth_refined, gt_depth[:, 1, 0, :, :], pose1,
                                                        intrinsic, intrinsic_inv)

            source2_extrinsic = gt_cam[:, 2, 0, :, :]
            pose2 = (source2_extrinsic @ torch.inverse(ref_extrinsic))[:, :3, :]
            warped_depth_loss_2 = get_warped_depth_loss(depth_refined, gt_depth[:, 2, 0, :, :], pose2,
                                                        intrinsic, intrinsic_inv)

            # if (epoch - start_epoch) < 2:
            #     loss_train = loss_idepth_refined
            # elif (epoch - start_epoch) < 4:
            #     loss_train = loss_idepth_refined + loss_depth_refined
            # elif (epoch - start_epoch) < 7:
            #     loss_train = loss_idepth_refined + loss_depth_refined + loss_normal_depth_refined
            # else:
            #     loss_train = loss_idepth_refined + loss_depth_refined + loss_normal_depth_refined + prob_map_loss

            loss_train += (warped_depth_loss_1 + warped_depth_loss_2)

            # Backward
            optimizer.zero_grad()
            loss_train.backward()
            # loss_idepth_refined.backward()
            optimizer.step()

            # update time
            batch_time.update(time.time() - tic)
            tic = time.time()

            if iter % cfg.print_interval == 0:

                _log.info(f"[{epoch:2d}][{iter:5d}/{len(data_loader):5d}] "
                          f"Time:  {batch_time.avg:.2f} "
                          f"Loss:  {loss_train.item():.4f} "
                          f"Depth  {loss_depth_1.item():.4f} "
                          f"Depth_warp1  {warped_depth_loss_1.item():.4f} "
                          f"Depth_warp2  {warped_depth_loss_2.item():.4f} "
                          f"LN:  {loss_normal_depth.item():.4f} "
                          f"IDepth:  {loss_idepth_1.item():.4f} "
                          f"Depth_refined:  {loss_depth_refined.item():.4f} "
                          f"LN_refined:  {loss_normal_depth_refined.item():.4f} "
                          f"IDepth_refined:  {loss_idepth_refined.item():.4f} "
                          f"prob_loss:  {prob_loss.item():.4f} "
                          f"prob_loss_depth:  {prob_loss_depth.item():.4f} "
                          f"prob_loss_minusmean:  {prob_loss_minusmean.item():.4f} "
                          f"prob_map_loss:  {prob_map_loss.item():.4f}"
                          )

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                with torch.no_grad():

                    # 1. Log scalar values (scalar summary)

                    info = {'loss': loss_train.item(),
                            'loss_idepth': loss_idepth_1.item(),
                            'loss_depth': loss_depth_1.item(),
                            'loss_normal_depth': loss_normal_depth.item(),
                            'loss_idepth_refined': loss_idepth_refined.item(),
                            'loss_depth_refined': loss_depth_refined.item(),
                            'loss_normal_depth_refined': loss_normal_depth_refined.item(),
                            'prob_loss': prob_loss.item(),
                            'prob_loss_depth': prob_loss_depth.item(),
                            'prob_loss_minusmean': prob_loss_minusmean.item(),
                            'prob_map_loss': prob_map_loss.item()}

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, global_step)

                    if iter % (cfg.print_interval * 10) == 0:
                        # 2. log histgrams
                        info = {'prob_map_gt': prob_map_gt.cpu().numpy(),
                                'prob_map': prob_map.cpu().numpy(),
                                'diff': np.clip(torch.abs(depth_refined - gt_depth).cpu().numpy(), a_min=0.0,
                                                a_max=8.0)}

                        for tag, values in info.items():
                            logger.histo_summary(tag, values, global_step)

                        # 3. Log training images (image summary)

                        gt_segmentation = gt_seg[:, 0, :, :]
                        gt_segmentation += 1
                        gt_segmentation[gt_segmentation == 21] = 0

                        info = {'rgb': image[:, 0, :, :, :].permute(0, 2, 3, 1).cpu().numpy(),
                                'gt_normal': gt_normal[:, 0, :, :, :].permute(0, 2, 3, 1).cpu().numpy(),
                                'normal_by_planes': normal_by_planes.permute(0, 2, 3, 1).cpu().numpy(),
                                'plane_normal': normals_from_plane_para[:, 0, :, :, :].permute(0, 2, 3,
                                                                                               1).cpu().numpy(),

                                'normal_from_depth_01': normal_from_depth_01.permute(0, 2, 3, 1).cpu().numpy(),
                                'normal_from_depth_refined': normal_from_depth_refined.permute(0, 2, 3,
                                                                                               1).cpu().numpy(),
                                'gt_seg': np.stack([colors[gt_segmentation.cpu().numpy(), 0],
                                                    colors[gt_segmentation.cpu().numpy(), 1],
                                                    colors[gt_segmentation.cpu().numpy(), 2]], axis=3),
                                'gt_idepth':
                                    np2Depth(gt_disparity[:, 0, :, :, :].squeeze(1).cpu().numpy()),
                                'pred_idepth_01':
                                    np2Depth(idepth_preds_01[0].squeeze(1).cpu().numpy()),
                                'pred_idepth_refined':
                                    np2Depth(idepth_refined.squeeze(1).cpu().numpy()),
                                'prob_map_pred': colorize_probmap(prob_map.squeeze(1).cpu().numpy()),
                                'prob_map_gt': colorize_probmap(prob_map_gt.squeeze(1).cpu().numpy())
                                }

                        for tag, images in info.items():
                            logger.image_summary(tag, images, global_step)
            # update global_step
            global_step = global_step + 1
            # except:
            #     print(sample['rgbs_filepath'])
            #     exit(1)
            if iter % (len(data_loader) // 8) == 0:
                # save checkpoint
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'depth_network_state_dict': depth_network.module.state_dict(),
                    'depth_refine_network_state_dict': depth_refine_network.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(checkpoint_dir, f"network_epoch_{epoch}_scale_{int(cfg.idepth_scale)}.pt"))


@ex.command
def train_wo_normal(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_dir = os.path.join('../experiments', str(_run._id), 'checkpoints')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    logger_dir = os.path.join('../experiments', str(_run._id), 'log')
    if not os.path.exists(logger_dir):
        os.mkdir(logger_dir)

    # logger
    logger = Logger(logger_dir)

    # build normal_network and depth_network
    depth_network = depthNet(idepth_scale=cfg.idepth_scale)
    # for p in depth_network.parameters():
    #     p.requires_grad = False

    depth_refine_network = DepthRefineNet(idepth_scale=cfg.idepth_scale)
    # set up optimizers
    network_params = list(depth_refine_network.parameters()) + list(depth_network.parameters())

    optimizer = get_optimizer(network_params, cfg.solver)

    if not cfg.resume_dir == 'None':
        print('resume training')
        checkpoint = torch.load(cfg.resume_dir)

        depth_network.load_state_dict(checkpoint['depth_network_state_dict'])

        try:
            depth_refine_network.load_state_dict(checkpoint['depth_refine_network_state_dict'])
        except:
            print("no checkpoint for refineNet")

        # optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']
        global_step = checkpoint['global_step']
    else:
        global_step = 0
        start_epoch = 0

    # load nets into gpu
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        gpu_num = torch.cuda.device_count()
        depth_network = torch.nn.DataParallel(depth_network)
        depth_refine_network = torch.nn.DataParallel(depth_refine_network)

    depth_network.to(device)
    depth_refine_network.to(device)

    # data loader
    data_loader = load_dataset('train', cfg.dataset)

    depth_network.train()
    depth_refine_network.train()

    criterion_234 = IdepthLoss_234()
    criterion_1 = IdepthLoss()
    criterion_idepth_prob = IdepthwithProbLoss()
    criterion_prob = ProbLoss()

    # main loop
    for epoch in range(start_epoch + 1, cfg.num_epochs):
        batch_time = AverageMeter()

        tic = time.time()
        for iter, sample in enumerate(data_loader):
            # try:
            image = sample['rgbs'].to(device)
            batch_size, views, c, h, w = image.shape
            instance = sample['plane_instance_segs']
            instance = instance.to(device)
            # semantic = sample['semantic'].to(device)
            gt_depth = sample['depths'].to(device)

            gt_seg = sample['plane_segs'].to(device)  # [b, views, h, w]
            normals_from_plane_para = sample['normals_from_plane_para'].to(device)

            gt_normal = sample['normals'].to(device)
            gt_normal_valid = gt_depth > 0.1
            plane_nums = sample['plane_nums']  # [b, views]
            # valid_region = sample['valid_region'].to(device)
            # gt_plane_instance_parameter = sample['plane_instance_parameter'].to(device)

            gt_cam = sample['cameras'].to(device)
            gt_disparity = sample['disparities'].to(device)

            idepth_preds_01, iconv_01 = depth_network(image[:, 0, :, :, :], image[:, 1, :, :, :],
                                                      gt_cam[:, 0, :, :, :], gt_cam[:, 1, :, :, :])
            idepth_preds_02, iconv_02 = depth_network(image[:, 0, :, :, :], image[:, 2, :, :, :],
                                                      gt_cam[:, 0, :, :, :], gt_cam[:, 2, :, :, :])

            ######################
            # left-right refine idepth
            ######################
            idepth_refined, prob_map = depth_refine_network(idepth01=idepth_preds_01[0],
                                                            idepth02=idepth_preds_02[0],
                                                            iconv01=iconv_01,
                                                            iconv02=iconv_02)

            loss_idepth_1 = (criterion_1(idepth_preds_01[0], gt_disparity[:, 0, :, :, :]) +
                             criterion_1(idepth_preds_02[0], gt_disparity[:, 0, :, :, :])) * 0.5

            loss_idepth_refined = criterion_1(idepth_refined, gt_disparity[:, 0, :, :, :])

            loss_idepth_234 = (criterion_234(idepth_preds_01, gt_disparity[:, 0, :, :, :]) +
                               criterion_234(idepth_preds_02, gt_disparity[:, 0, :, :, :])) * 0.5

            eps = 1e-8
            depth_preds_01 = torch.div(1.0, idepth_preds_01[0].squeeze(1) + eps)
            depth_preds_02 = torch.div(1.0, idepth_preds_02[0].squeeze(1) + eps)

            depth_refined = torch.div(1.0, idepth_refined.squeeze(1) + eps)

            ####################################
            # prob loss
            #####################################
            prob_loss_depth = criterion_idepth_prob(idepth_refined, gt_disparity[:, 0, :, :, :], prob_map) + \
                              criterion_idepth_prob(depth_refined.unsqueeze(1), gt_depth[:, 0, :, :, :], prob_map)
            prob_loss_minusmean = 1 - prob_map.mean()

            prob_map_loss, prob_map_gt = criterion_prob(prob_map, idepth_refined, gt_disparity[:, 0, :, :, :])

            prob_loss = 5 * prob_loss_depth + prob_loss_minusmean  # + prob_map_loss

            intrinsic = gt_cam[:, 0, 1, 0:3, 0:3]
            intrinsic_inv = torch.inverse(intrinsic)

            loss_depth_1 = (criterion_1(depth_preds_01.unsqueeze(1), gt_depth[:, 0, :, :, :]) +
                            criterion_1(depth_preds_02.unsqueeze(1), gt_depth[:, 0, :, :, :])) * 0.5

            loss_depth_refined = criterion_1(depth_refined.unsqueeze(1), gt_depth[:, 0, :, :, :])

            if (epoch - start_epoch) < 5:
                loss_train = loss_idepth_1 + loss_idepth_234 + loss_idepth_refined
            else:
                loss_train = loss_depth_1 + loss_depth_refined + \
                             (loss_idepth_1 + loss_idepth_234 + loss_idepth_refined) + prob_loss

            # Backward
            optimizer.zero_grad()
            loss_train.backward()
            # loss_idepth_refined.backward()
            optimizer.step()

            # update time
            batch_time.update(time.time() - tic)
            tic = time.time()

            if iter % cfg.print_interval == 0:

                _log.info(f"[{epoch:2d}][{iter:5d}/{len(data_loader):5d}] "
                          f"Time:  {batch_time.avg:.2f} "
                          f"Loss:  {loss_train.item():.4f} "
                          f"Depth  {loss_depth_1.item():.4f} "
                          f"IDepth:  {loss_idepth_1.item():.4f} "
                          f"Depth_refined:  {loss_depth_refined.item():.4f} "
                          f"IDepth_refined:  {loss_idepth_refined.item():.4f} "
                          f"prob_loss:  {prob_loss.item():.4f} "
                          f"prob_loss_depth:  {prob_loss_depth.item():.4f} "
                          f"prob_loss_minusmean:  {prob_loss_minusmean.item():.4f} "
                          f"prob_map_loss:  {prob_map_loss.item():.4f}"
                          )

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                with torch.no_grad():

                    # 1. Log scalar values (scalar summary)

                    info = {'loss': loss_train.item(),
                            'loss_idepth': loss_idepth_1.item(),
                            'loss_depth': loss_depth_1.item(),
                            'loss_idepth_refined': loss_idepth_refined.item(),
                            'loss_depth_refined': loss_depth_refined.item(),
                            'prob_loss': prob_loss.item(),
                            'prob_loss_depth': prob_loss_depth.item(),
                            'prob_loss_minusmean': prob_loss_minusmean.item(),
                            'prob_map_loss': prob_map_loss.item()
                            }

                    for tag, value in info.items():
                        logger.scalar_summary(tag, value, global_step)

                    if iter % (cfg.print_interval * 10) == 0:
                        # 2. log histgrams
                        info = {'prob_map_gt': prob_map_gt.cpu().numpy(),
                                'prob_map': prob_map.cpu().numpy(),
                                'diff': np.clip(torch.abs(depth_refined - gt_depth).cpu().numpy(), a_min=0.0,
                                                a_max=8.0)}

                        for tag, values in info.items():
                            logger.histo_summary(tag, values, global_step)

                        # 3. Log training images (image summary)

                        gt_segmentation = gt_seg[:, 0, :, :]
                        gt_segmentation += 1
                        gt_segmentation[gt_segmentation == 21] = 0

                        info = {'rgb': image[:, 0, :, :, :].permute(0, 2, 3, 1).cpu().numpy(),
                                'gt_normal': gt_normal[:, 0, :, :, :].permute(0, 2, 3, 1).cpu().numpy(),
                                'plane_normal': normals_from_plane_para[:, 0, :, :, :].permute(0, 2, 3,
                                                                                               1).cpu().numpy(),
                                'gt_seg': np.stack([colors[gt_segmentation.cpu().numpy(), 0],
                                                    colors[gt_segmentation.cpu().numpy(), 1],
                                                    colors[gt_segmentation.cpu().numpy(), 2]], axis=3),
                                'gt_idepth':
                                    np2Depth(gt_disparity[:, 0, :, :, :].squeeze(1).cpu().numpy()),
                                'pred_idepth_01':
                                    np2Depth(idepth_preds_01[0].squeeze(1).cpu().numpy()),
                                'pred_idepth_refined':
                                    np2Depth(idepth_refined.squeeze(1).cpu().numpy()),
                                'prob_map_pred': colorize_probmap(prob_map.squeeze(1).cpu().numpy()),
                                'prob_map_gt': colorize_probmap(prob_map_gt.squeeze(1).cpu().numpy())
                                }

                        for tag, images in info.items():
                            logger.image_summary(tag, images, global_step)
            # update global_step
            global_step = global_step + 1
            # except:
            #     print(sample['rgbs_filepath'])
            #     exit(1)
            if iter % (len(data_loader) // 8) == 0:
                # save checkpoint
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'depth_network_state_dict': depth_network.module.state_dict(),
                    'depth_refine_network_state_dict': depth_refine_network.module.state_dict(),
                    'optimizer': optimizer.state_dict()},
                    os.path.join(checkpoint_dir, f"network_epoch_{epoch}_scale_{int(cfg.idepth_scale)}.pt"))




if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore", category=DeprecationWarning)

    assert LooseVersion(torch.__version__) >= LooseVersion('1.2.0'), \
        'PyTorch>=1.2.0 is required, the used version is ' + torch.__version__

    ex.add_config('./configs/config_unet_mean_shift.yaml')
    ex.run_commandline()
