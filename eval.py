from utils.metric import *
import argparse
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from scannet.preprocess import *
import scipy
import os
import cv2
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver
from easydict import EasyDict as edict

from utils.disp import *
from depthnet.depthNet_model import depthNet, DepthRefineNet
from depthnet.depth_util import np2Depth, Depth2normal, colorize_probmap, normal2color, depth2color

import time

from utils.disp import colors_256 as colors

ex = Experiment()
ex.observers.append(FileStorageObserver.create('../evaluations_7_scenes'))


class LoadSevenScenes(object):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.test_seqs_list = [('chess', 'seq-03'),
                               ('chess', 'seq-05'),
                               ('fire', 'seq-03'),
                               ('fire', 'seq-04'),
                               ('heads', 'seq-01'),
                               ('office', 'seq-02'),
                               ('office', 'seq-06'),
                               ('office', 'seq-07'),
                               ('office', 'seq-09'),
                               ('pumpkin', 'seq-01'),
                               ('pumpkin', 'seq-07'),
                               ('redkitchen', 'seq-03'),
                               ('redkitchen', 'seq-04'),
                               ('redkitchen', 'seq-06'),
                               ('redkitchen', 'seq-12'),
                               ('redkitchen', 'seq-14'),
                               ('stairs', 'seq-01'),
                               ('stairs', 'seq-04')]

        self.intrinsics = np.asarray([[585, 0, 320],
                                      [0, 585, 240],
                                      [0, 0, 1]])

    def get_filepaths(self, scene, seq):
        """ load list of filenames from one seq of scene;
        return filepaths of rgbs, depths, poses
        """
        seq_dir = os.path.join(self.root_dir, scene, seq)
        filepaths_list = []
        for filename in sorted(os.listdir(seq_dir)):
            if "color" in filename:
                rgb_name = filename
                depth_name = rgb_name.replace("color", "depth")
                pose_name = rgb_name.replace("color.png", "pose.txt")
                pred_depth_name = rgb_name.replace("color", "pred_depth")
                sample_path = {'rgb': os.path.join(seq_dir, rgb_name),
                               'depth': os.path.join(seq_dir, depth_name),
                               'pose': os.path.join(seq_dir, pose_name),
                               'pred_depth_name': pred_depth_name}
                filepaths_list.append(sample_path)
        return filepaths_list

    def load_sample(self, sample_path, image_height_expected, image_width_expected):
        rgb = cv2.imread(sample_path['rgb'], -1)
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        depth = cv2.imread(sample_path['depth'], -1) / 1000.0
        pose = self.read_pose_file(sample_path['pose'])

        extrinsics = self.pose2extrinsics(pose)
        cam = self.get_cam(self.intrinsics, extrinsics)

        original_image_width = rgb.shape[1]
        original_image_height = rgb.shape[0]
        scale_x = float(image_width_expected) / original_image_width
        scale_y = float(image_height_expected) / original_image_height

        cam = self.scale_cam(cam, scale_x, scale_y)
        rgb = self.scale_img(rgb, image_height_expected, image_width_expected, 'linear')
        rgb = self.normalize_image(rgb)
        # depth = self.scale_img(depth, image_height_expected, image_width_expected, 'nearest')

        rgb, depth, cam = self.toTensor(rgb, depth, cam)

        return rgb, depth, cam, sample_path['pred_depth_name']

    def toTensor(self, rgb, depth, cam):
        rgb = torch.from_numpy(np.float32(rgb))
        rgb = rgb.permute(2, 0, 1)  # (h,w,c) to (c, h, w)

        depth = torch.from_numpy(np.float32(depth))

        cam = torch.from_numpy(np.float32(cam))

        return rgb, depth, cam

    def scale_img(self, image, expected_height, expected_width, interpolation):
        # although opencv load image in shape (height, width, channel), cv2.resize still need shape (width, height)
        if interpolation == 'linear':
            return cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
        if interpolation == 'nearest':
            return cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_NEAREST)
        if interpolation is None:
            raise Exception('interpolation cannot be None')

    def normalize_image(self, img):
        """
        Zero mean and Unit variance normalization to input image
        :param img: input image
        :return: normalized image
        """
        img = img / 255.
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        img_normal = (img - mean) / std
        return img_normal.astype(np.float32)

    def read_pose_file(self, filepath):
        """Read in the pose/*.txt and parse into a ndarray"""
        pose = np.loadtxt(filepath, dtype='f', delimiter='\t ')

        return pose

    def pose2extrinsics(self, pose):
        """Convert pose(camera to world) to extrinsic matrix(world to camera)"""
        extrinsics = np.linalg.inv(pose)
        return extrinsics

    def get_cam(self, intrinsics, extrinsics):
        """convert to cam"""
        cam = np.zeros((2, 4, 4))

        # read extrinsic(world to camera) or pose(camera to pose)
        cam[0] = extrinsics

        # read intrinsic
        cam[1][0:3, 0:3] = self.intrinsics

        return cam

    def scale_cam(self, cam, scale_x, scale_y):
        """scale the camera intrinsics of one view"""
        new_cam = copy(cam)
        # focal
        new_cam[1][0][0] *= scale_x
        new_cam[1][1][1] *= scale_y

        # principle point:
        new_cam[1][0][2] *= scale_x
        new_cam[1][1][2] *= scale_y

        return new_cam


@ex.command
def eval(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluation_dir = os.path.join('../evaluations_7_scenes_eval', str(_run._id))
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    # build normal_network and depth_network
    depth_network = depthNet(idepth_scale=cfg.idepth_scale)

    if not cfg.resume_dir == 'None':
        print('resume training')
        checkpoint = torch.load(cfg.resume_dir)
        # should change to here this line

        try:
            try:
                depth_network.load_state_dict(checkpoint['depth_network_state_dict'])
            except:
                # for model is saved by nn.DataParallel
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                state_dict = checkpoint['depth_network_state_dict']
                for k, v in state_dict.items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # load params
                depth_network.load_state_dict(new_state_dict)
        except:
            depth_network.load_state_dict(checkpoint['state_dict'])

        depth_network = depth_network.to(device)

    else:
        print("evaluation must need checkpoint")

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        depth_network = torch.nn.DataParallel(depth_network)

    if cfg.resume_dir == 'None':
        depth_network.to(device)

    depth2normal = Depth2normal(cfg.k_size)
    depth2normal.to(device)

    # data loader
    sevenScenes = LoadSevenScenes(cfg.dataset.root_dir)

    depth_network.eval()

    # main loop

    count = 0
    total_time = 0
    for scene, seq in sevenScenes.test_seqs_list:

        rgb_dir = os.path.join(evaluation_dir, scene, seq, 'rgb')
        gt_depth_dir = os.path.join(evaluation_dir, scene, seq, 'gt_depth')
        pred_depth_dir = os.path.join(evaluation_dir, scene, seq, 'pred_depth')
        pred_normal_dir = os.path.join(evaluation_dir, scene, seq, 'pred_normal')

        dirs = [rgb_dir, gt_depth_dir, pred_depth_dir, pred_normal_dir]

        for dir in dirs:
            if not os.path.exists(dir):
                os.makedirs(dir)

        filepaths_list = sevenScenes.get_filepaths(scene, seq)

        for index in range(0, len(filepaths_list) - 10, 1):
            if index % 10 != 0:
                continue
            count += 1
            print(scene, seq, index)
            ref_sample_path = filepaths_list[index]
            source_sample_path = filepaths_list[index + 10]
            ref_rgb, gt_depth, ref_cam, pred_depth_name = sevenScenes.load_sample(ref_sample_path,
                                                                                  cfg.dataset.image_height,
                                                                                  cfg.dataset.image_width)
            source_rgb, _, source_cam, _ = sevenScenes.load_sample(source_sample_path, cfg.dataset.image_height,
                                                                   cfg.dataset.image_width)
            ref_rgb = ref_rgb.to(device)
            source_rgb = source_rgb.to(device)
            c, h, w = ref_rgb.shape

            gt_depth = gt_depth.to(device)  # [h, w]

            ref_cam = ref_cam.to(device)
            source_cam = source_cam.to(device)
            start_time = time.monotonic()
            idepth_preds, _ = depth_network(ref_rgb.unsqueeze(0), source_rgb.unsqueeze(0),
                                            ref_cam.unsqueeze(0), source_cam.unsqueeze(0))

            end_time = time.monotonic()
            time_interval = (end_time - start_time)
            total_time += time_interval

            # print("time:", (total_time / count))

            depth_preds = torch.div(1.0, idepth_preds[0].squeeze(1))
            intrinsic = ref_cam[1, 0:3, 0:3]
            intrinsic_inv = torch.inverse(intrinsic)
            normal_from_depth, _ = depth2normal(depth_preds, intrinsic_inv.unsqueeze(0))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #
            with torch.no_grad():
                info = {'rgb': ref_rgb.permute(1, 2, 0).cpu().numpy(),
                        'normal_from_depth': normal_from_depth.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
                        'gt_depth':
                            gt_depth.cpu().numpy(),
                        'pred_idepth':
                            idepth_preds[0].squeeze().cpu().numpy()
                        }

                rgb_filepath = os.path.join(rgb_dir, pred_depth_name.replace("pred_depth", "color"))
                scipy.misc.imsave(rgb_filepath, info['rgb'])

                pred_normal_filepath = os.path.join(pred_normal_dir,
                                                    pred_depth_name.replace("pred_depth.png", "pred_normal.npy"))
                np.save(pred_normal_filepath, info['normal_from_depth'])

                pred_normal_color = normal2color(info['normal_from_depth'])
                pred_normal_color_filepath = os.path.join(pred_normal_dir,
                                                          pred_depth_name.replace(
                                                              "pred_depth",
                                                              "pred_normal"))
                scipy.misc.imsave(pred_normal_color_filepath, pred_normal_color)

                gt_depth_filepath = os.path.join(gt_depth_dir,
                                                 pred_depth_name.replace("pred_depth.png", "gt_depth.npy"))
                np.save(gt_depth_filepath, info['gt_depth'])

                gt_depth_color = depth2color(info['gt_depth'])
                gt_depth_color_filepath = os.path.join(gt_depth_dir,
                                                       pred_depth_name.replace("pred_depth", "gt_depth"))
                scipy.misc.imsave(gt_depth_color_filepath, gt_depth_color)

                pred_depth = np.reciprocal(info['pred_idepth'] + 1e-4)
                pred_depth[pred_depth > 100] = 0
                pred_depth_filepath = os.path.join(pred_depth_dir,
                                                   pred_depth_name.replace("pred_depth.png", "pred_depth.npy"))
                np.save(pred_depth_filepath, pred_depth)

                pred_depth_color = depth2color(pred_depth)
                pred_depth_color_filepath = os.path.join(pred_depth_dir, pred_depth_name)
                scipy.misc.imsave(pred_depth_color_filepath, pred_depth_color)


@ex.command
def eval_refine(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluation_dir = os.path.join('../evaluations_7_scenes_refine', str(_run._id))
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    # build normal_network and depth_network
    depth_network = depthNet(idepth_scale=cfg.idepth_scale)
    depth_refine_network = DepthRefineNet(idepth_scale=cfg.idepth_scale)

    if not cfg.resume_dir == 'None':
        print('resume training')
        checkpoint = torch.load(cfg.resume_dir)
        # should change to here this line

        try:
            depth_network.load_state_dict(checkpoint['depth_network_state_dict'])
        except:
            # for model is saved by nn.DataParallel
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            state_dict = checkpoint['depth_network_state_dict']
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            depth_network.load_state_dict(new_state_dict)

        try:
            depth_refine_network.load_state_dict(checkpoint['depth_refine_network_state_dict'])
        except:
            # for model is saved by nn.DataParallel
            from collections import OrderedDict
            refine_state_dict = OrderedDict()
            state_dict = checkpoint['depth_refine_network_state_dict']
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                refine_state_dict[name] = v
            # load params
            depth_refine_network.load_state_dict(refine_state_dict)

    else:
        print("evaluation must need checkpoint")

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        depth_network = torch.nn.DataParallel(depth_network)

    depth_network.to(device)
    depth_refine_network.to(device)

    depth2normal = Depth2normal(cfg.k_size)
    depth2normal.to(device)

    # data loader
    sevenScenes = LoadSevenScenes(cfg.dataset.root_dir)

    depth_network.eval()
    depth_refine_network.eval()

    # main loop
    count = 0
    total_time = 0
    with torch.no_grad():
        for scene, seq in sevenScenes.test_seqs_list:

            rgb_dir = os.path.join(evaluation_dir, scene, seq, 'rgb')
            gt_depth_dir = os.path.join(evaluation_dir, scene, seq, 'gt_depth')
            pred_depth_dir = os.path.join(evaluation_dir, scene, seq, 'pred_depth')
            pred_normal_dir = os.path.join(evaluation_dir, scene, seq, 'pred_normal')
            prob_map_dir = os.path.join(evaluation_dir, scene, seq, 'prob_map')

            dirs = [rgb_dir, gt_depth_dir, pred_depth_dir, pred_normal_dir, prob_map_dir]

            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            filepaths_list = sevenScenes.get_filepaths(scene, seq)

            for index in range(10, len(filepaths_list) - 10, 1):
                if index % 3 != 0:
                    continue
                count += 1
                print(scene, seq, index)
                ref_sample_path = filepaths_list[index]
                source_1_sample_path = filepaths_list[index + 10]
                source_2_sample_path = filepaths_list[index - 10]
                ref_rgb, gt_depth, ref_cam, pred_depth_name = sevenScenes.load_sample(ref_sample_path,
                                                                                      cfg.dataset.image_height,
                                                                                      cfg.dataset.image_width)
                source_1_rgb, _, source_1_cam, _ = sevenScenes.load_sample(source_1_sample_path,
                                                                           cfg.dataset.image_height,
                                                                           cfg.dataset.image_width)
                ref_rgb = ref_rgb.to(device)
                source_1_rgb = source_1_rgb.to(device)

                source_2_rgb, _, source_2_cam, _ = sevenScenes.load_sample(source_2_sample_path,
                                                                           cfg.dataset.image_height,
                                                                           cfg.dataset.image_width)
                ref_rgb = ref_rgb.to(device)
                source_2_rgb = source_2_rgb.to(device)

                c, h, w = ref_rgb.shape

                gt_depth = gt_depth.to(device)  # [h, w]

                ref_cam = ref_cam.to(device)
                source_1_cam = source_1_cam.to(device)
                source_2_cam = source_2_cam.to(device)

                start_time = time.monotonic()
                idepth_preds_01, iconv_01 = depth_network(ref_rgb.unsqueeze(0), source_1_rgb.unsqueeze(0),
                                                          ref_cam.unsqueeze(0), source_1_cam.unsqueeze(0))
                idepth_preds_02, iconv_02 = depth_network(ref_rgb.unsqueeze(0), source_2_rgb.unsqueeze(0),
                                                          ref_cam.unsqueeze(0), source_2_cam.unsqueeze(0))
                idepth_refined, prob_map = depth_refine_network(idepth01=idepth_preds_01[0],
                                                                idepth02=idepth_preds_02[0],
                                                                iconv01=iconv_01,
                                                                iconv02=iconv_02)
                end_time = time.monotonic()
                time_interval = (end_time - start_time)
                total_time += time_interval

                depth_preds = torch.div(1.0, idepth_refined.squeeze(1))
                intrinsic = ref_cam[1, 0:3, 0:3]
                intrinsic_inv = torch.inverse(intrinsic)
                normal_from_depth, _ = depth2normal(depth_preds, intrinsic_inv.unsqueeze(0))

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #

                info = {'rgb': ref_rgb.permute(1, 2, 0).cpu().numpy(),
                        'normal_from_depth': normal_from_depth.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
                        'gt_depth':
                            gt_depth.cpu().numpy(),
                        'pred_idepth':
                            idepth_refined.squeeze().cpu().numpy(),
                        'prob_map': prob_map.squeeze().cpu().numpy()
                        }

                rgb_filepath = os.path.join(rgb_dir, pred_depth_name.replace("pred_depth", "color"))
                scipy.misc.imsave(rgb_filepath, info['rgb'])

                pred_normal_filepath = os.path.join(pred_normal_dir,
                                                    pred_depth_name.replace("pred_depth.png", "pred_normal.npy"))
                np.save(pred_normal_filepath, info['normal_from_depth'])

                pred_normal_color = normal2color(info['normal_from_depth'])
                pred_normal_color_filepath = os.path.join(pred_normal_dir,
                                                          pred_depth_name.replace(
                                                              "pred_depth",
                                                              "pred_normal"))
                scipy.misc.imsave(pred_normal_color_filepath, pred_normal_color)

                gt_depth_filepath = os.path.join(gt_depth_dir,
                                                 pred_depth_name.replace("pred_depth.png", "gt_depth.npy"))
                np.save(gt_depth_filepath, info['gt_depth'])

                gt_depth_color = depth2color(info['gt_depth'])
                gt_depth_color_filepath = os.path.join(gt_depth_dir,
                                                       pred_depth_name.replace("pred_depth", "gt_depth"))
                scipy.misc.imsave(gt_depth_color_filepath, gt_depth_color)

                pred_depth = np.reciprocal(info['pred_idepth'] + 1e-4)
                pred_depth[pred_depth > 100] = 0
                pred_depth_filepath = os.path.join(pred_depth_dir,
                                                   pred_depth_name.replace("pred_depth.png", "pred_depth.npy"))
                np.save(pred_depth_filepath, pred_depth)

                pred_depth_color = depth2color(pred_depth)
                pred_depth_color_filepath = os.path.join(pred_depth_dir, pred_depth_name)
                scipy.misc.imsave(pred_depth_color_filepath, pred_depth_color)

                prob_map_color = colorize_probmap(info['prob_map'])
                prob_map_color_filepath = os.path.join(prob_map_dir,
                                                       pred_depth_name.replace("pred_depth.png", "prob_map.png"))
                scipy.misc.imsave(prob_map_color_filepath, prob_map_color)

                prob_map_filepath = os.path.join(prob_map_dir,
                                                 pred_depth_name.replace("pred_depth.png", "prob_map.npy"))
                np.save(prob_map_filepath, info['prob_map'])


@ex.command
def eval_refine_five_views(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluation_dir = os.path.join('../evaluations_7_scenes_refine_five_views', str(_run._id))
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    # build normal_network and depth_network
    depth_network = depthNet(idepth_scale=cfg.idepth_scale)
    depth_refine_network = DepthRefineNet(idepth_scale=cfg.idepth_scale)

    if not cfg.resume_dir == 'None':
        print('resume training')
        checkpoint = torch.load(cfg.resume_dir)
        # should change to here this line

        depth_network.load_state_dict(checkpoint['depth_network_state_dict'])
        depth_network.to(device)

        depth_refine_network.load_state_dict(checkpoint['depth_refine_network_state_dict'])
        depth_refine_network.to(device)

    else:
        print("evaluation must need checkpoint")

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        depth_network = torch.nn.DataParallel(depth_network)

    if cfg.resume_dir == 'None':
        depth_network.to(device)

    depth2normal = Depth2normal(cfg.k_size)
    depth2normal.to(device)

    # data loader
    sevenScenes = LoadSevenScenes(cfg.dataset.root_dir)

    depth_network.eval()
    depth_refine_network.eval()

    with torch.no_grad():
        # main loop
        count = 0
        total_time = 0
        for scene, seq in sevenScenes.test_seqs_list[0:1]:

            rgb_dir = os.path.join(evaluation_dir, scene, seq, 'rgb')
            gt_depth_dir = os.path.join(evaluation_dir, scene, seq, 'gt_depth')
            pred_depth_dir = os.path.join(evaluation_dir, scene, seq, 'pred_depth')
            pred_normal_dir = os.path.join(evaluation_dir, scene, seq, 'pred_normal')
            prob_map_dir = os.path.join(evaluation_dir, scene, seq, 'prob_map')

            dirs = [rgb_dir, gt_depth_dir, pred_depth_dir, pred_normal_dir, prob_map_dir]

            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            filepaths_list = sevenScenes.get_filepaths(scene, seq)

            for index in range(10, len(filepaths_list) - 20, 1):
                if index % 3 != 0:
                    continue
                count += 1
                print(scene, seq, index)
                ref_sample_path = filepaths_list[index]
                ## expriments show that
                # (10, 5, 0, -5, -10) is better than (15, 10, 0, 10, 15)
                source_1_sample_path = filepaths_list[index + 10]
                source_2_sample_path = filepaths_list[index - 10]
                source_3_sample_path = filepaths_list[index + 5]
                source_4_sample_path = filepaths_list[index - 5]

                # camera may be invalid
                try:
                    ref_rgb, gt_depth, ref_cam, pred_depth_name = sevenScenes.load_sample(ref_sample_path,
                                                                                          cfg.dataset.image_height,
                                                                                          cfg.dataset.image_width)
                    source_1_rgb, _, source_1_cam, _ = sevenScenes.load_sample(source_1_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_2_rgb, _, source_2_cam, _ = sevenScenes.load_sample(source_2_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_3_rgb, _, source_3_cam, _ = sevenScenes.load_sample(source_3_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_4_rgb, _, source_4_cam, _ = sevenScenes.load_sample(source_4_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                except:
                    print("invalid_camera")
                    continue

                ref_rgb = ref_rgb.to(device)
                source_1_rgb = source_1_rgb.to(device)
                source_2_rgb = source_2_rgb.to(device)
                source_3_rgb = source_3_rgb.to(device)
                source_4_rgb = source_4_rgb.to(device)

                c, h, w = ref_rgb.shape

                gt_depth = gt_depth.to(device)  # [h, w]

                ref_cam = ref_cam.to(device)
                source_1_cam = source_1_cam.to(device)
                source_2_cam = source_2_cam.to(device)
                source_3_cam = source_3_cam.to(device)
                source_4_cam = source_4_cam.to(device)

                ref_rgb_batch = ref_rgb.unsqueeze(0).repeat(4, 1, 1, 1)
                source_rgb_batch = torch.cat([source_1_rgb.unsqueeze(0), source_2_rgb.unsqueeze(0),
                                              source_3_rgb.unsqueeze(0), source_4_rgb.unsqueeze(0)], dim=0)
                ref_cam_batch = torch.cat([ref_cam.unsqueeze(0), ref_cam.unsqueeze(0),
                                           ref_cam.unsqueeze(0), ref_cam.unsqueeze(0)], dim=0)
                source_cam_batch = torch.cat([source_1_cam.unsqueeze(0), source_2_cam.unsqueeze(0),
                                              source_3_cam.unsqueeze(0), source_4_cam.unsqueeze(0)], dim=0)

                start_time = time.monotonic()
                # idepth_preds_01, iconv_01 = depth_network(ref_rgb.unsqueeze(0), source_1_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_1_cam.unsqueeze(0))
                # idepth_preds_02, iconv_02 = depth_network(ref_rgb.unsqueeze(0), source_2_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_2_cam.unsqueeze(0))
                # idepth_preds_03, iconv_03 = depth_network(ref_rgb.unsqueeze(0), source_3_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_3_cam.unsqueeze(0))
                # idepth_preds_04, iconv_04 = depth_network(ref_rgb.unsqueeze(0), source_4_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_4_cam.unsqueeze(0))
                # idepth_refined, prob_map = depth_refine_network(idepth01=(idepth_preds_01[0] + idepth_preds_03[0]) * 0.5,
                #                                                 idepth02=(idepth_preds_02[0] + idepth_preds_04[0]) * 0.5,
                #                                                 iconv01=(iconv_01 + iconv_03) * 0.5,
                #                                                 iconv02=(iconv_02 + iconv_04) * 0.5)
                idepth_preds_01, iconv_01 = depth_network(ref_rgb_batch, source_rgb_batch,
                                                          ref_cam_batch, source_cam_batch)

                idepth_refined, prob_map = depth_refine_network(
                    idepth01=(idepth_preds_01[0][0:1] + idepth_preds_01[0][2:3]) * 0.5,
                    idepth02=(idepth_preds_01[0][1:2] + idepth_preds_01[0][3:4]) * 0.5,
                    iconv01=(iconv_01[0:1] + iconv_01[2:3]) * 0.5,
                    iconv02=(iconv_01[1:2] + iconv_01[3:4]) * 0.5)
                end_time = time.monotonic()
                time_interval = (end_time - start_time)
                total_time += time_interval

            print("time:", (total_time / count))
            # depth_preds = torch.div(1.0, idepth_refined.squeeze(1))
            # intrinsic = ref_cam[1, 0:3, 0:3]
            # intrinsic_inv = torch.inverse(intrinsic)
            # normal_from_depth, _ = depth2normal(depth_preds, intrinsic_inv.unsqueeze(0))
            #
            # # ================================================================== #
            # #                        Tensorboard Logging                         #
            # # ================================================================== #
            # with torch.no_grad():
            #     info = {'rgb': ref_rgb.permute(1, 2, 0).cpu().numpy(),
            #             'normal_from_depth': normal_from_depth.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
            #             'gt_depth':
            #                 gt_depth.cpu().numpy(),
            #             'pred_idepth':
            #                 idepth_refined.squeeze().cpu().numpy(),
            #             'prob_map': prob_map.squeeze().cpu().numpy()
            #             }
            #
            #     rgb_filepath = os.path.join(rgb_dir, pred_depth_name.replace("pred_depth", "color"))
            #     scipy.misc.imsave(rgb_filepath, info['rgb'])
            #
            #     pred_normal_filepath = os.path.join(pred_normal_dir,
            #                                         pred_depth_name.replace("pred_depth.png", "pred_normal.npy"))
            #     np.save(pred_normal_filepath, info['normal_from_depth'])
            #
            #     pred_normal_color = normal2color(info['normal_from_depth'])
            #     pred_normal_color_filepath = os.path.join(pred_normal_dir,
            #                                               pred_depth_name.replace(
            #                                                   "pred_depth",
            #                                                   "pred_normal"))
            #     scipy.misc.imsave(pred_normal_color_filepath, pred_normal_color)
            #
            #     gt_depth_filepath = os.path.join(gt_depth_dir,
            #                                      pred_depth_name.replace("pred_depth.png", "gt_depth.npy"))
            #     np.save(gt_depth_filepath, info['gt_depth'])
            #
            #     gt_depth_color = depth2color(info['gt_depth'])
            #     gt_depth_color_filepath = os.path.join(gt_depth_dir,
            #                                            pred_depth_name.replace("pred_depth", "gt_depth"))
            #     scipy.misc.imsave(gt_depth_color_filepath, gt_depth_color)
            #
            #     pred_depth = np.reciprocal(info['pred_idepth'] + 1e-4)
            #     pred_depth[pred_depth > 100] = 0
            #     pred_depth_filepath = os.path.join(pred_depth_dir,
            #                                        pred_depth_name.replace("pred_depth.png", "pred_depth.npy"))
            #     np.save(pred_depth_filepath, pred_depth)
            #
            #     pred_depth_color = depth2color(pred_depth)
            #     pred_depth_color_filepath = os.path.join(pred_depth_dir, pred_depth_name)
            #     scipy.misc.imsave(pred_depth_color_filepath, pred_depth_color)
            #
            #     prob_map_color = colorize_probmap(info['prob_map'])
            #     prob_map_color_filepath = os.path.join(prob_map_dir,
            #                                            pred_depth_name.replace("pred_depth.png", "prob_map.png"))
            #     scipy.misc.imsave(prob_map_color_filepath, prob_map_color)
            #
            #     prob_map_filepath = os.path.join(prob_map_dir,
            #                                      pred_depth_name.replace("pred_depth.png", "prob_map.npy"))
            #     np.save(prob_map_filepath, info['prob_map'])


@ex.command
def eval_refine_seven_views(_run, _log):
    cfg = edict(_run.config)

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    random.seed(cfg.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluation_dir = os.path.join('../evaluations_7_scenes_refine_seven_views', str(_run._id))
    if not os.path.exists(evaluation_dir):
        os.makedirs(evaluation_dir)

    # build normal_network and depth_network
    depth_network = depthNet(idepth_scale=cfg.idepth_scale)
    depth_refine_network = DepthRefineNet(idepth_scale=cfg.idepth_scale)

    if not cfg.resume_dir == 'None':
        print('resume training')
        checkpoint = torch.load(cfg.resume_dir)
        # should change to here this line

        try:
            depth_network.load_state_dict(checkpoint['depth_network_state_dict'])
        except:
            # for model is saved by nn.DataParallel
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            state_dict = checkpoint['depth_network_state_dict']
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            # load params
            depth_network.load_state_dict(new_state_dict)

        try:
            depth_refine_network.load_state_dict(checkpoint['depth_refine_network_state_dict'])
        except:
            # for model is saved by nn.DataParallel
            from collections import OrderedDict
            refine_state_dict = OrderedDict()
            state_dict = checkpoint['depth_refine_network_state_dict']
            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                refine_state_dict[name] = v
            # load params
            depth_refine_network.load_state_dict(refine_state_dict)

    else:
        print("evaluation must need checkpoint")

    # load nets into gpu
    if cfg.num_gpus > 1 and torch.cuda.is_available():
        depth_network = torch.nn.DataParallel(depth_network)

    depth2normal = Depth2normal(cfg.k_size)
    depth2normal.to(device)
    depth_network.to(device)
    depth_refine_network.to(device)

    # data loader
    sevenScenes = LoadSevenScenes(cfg.dataset.root_dir)

    depth_network.eval()
    depth_refine_network.eval()

    # main loop
    count = 0.0
    total_time = 0.0
    with torch.no_grad():
        for scene, seq in sevenScenes.test_seqs_list:

            rgb_dir = os.path.join(evaluation_dir, scene, seq, 'rgb')
            gt_depth_dir = os.path.join(evaluation_dir, scene, seq, 'gt_depth')
            pred_depth_dir = os.path.join(evaluation_dir, scene, seq, 'pred_depth')
            pred_normal_dir = os.path.join(evaluation_dir, scene, seq, 'pred_normal')
            prob_map_dir = os.path.join(evaluation_dir, scene, seq, 'prob_map')

            dirs = [rgb_dir, gt_depth_dir, pred_depth_dir, pred_normal_dir, prob_map_dir]

            for dir in dirs:
                if not os.path.exists(dir):
                    os.makedirs(dir)

            filepaths_list = sevenScenes.get_filepaths(scene, seq)

            for index in range(10, len(filepaths_list) - 20, 1):
                if index % 9 != 0:
                    continue
                print(scene, seq, index)
                count += 1
                ref_sample_path = filepaths_list[index]
                ## expriments show that
                # (20, 10, 5, 0, -5, -10, -20) is best
                source_1_sample_path = filepaths_list[index + 10]
                source_2_sample_path = filepaths_list[index - 10]
                source_3_sample_path = filepaths_list[index + 5]
                source_4_sample_path = filepaths_list[index - 5]
                source_5_sample_path = filepaths_list[index + 20]
                source_6_sample_path = filepaths_list[index - 20]

                # camera may be invalid
                try:
                    ref_rgb, gt_depth, ref_cam, pred_depth_name = sevenScenes.load_sample(ref_sample_path,
                                                                                          cfg.dataset.image_height,
                                                                                          cfg.dataset.image_width)
                    source_1_rgb, _, source_1_cam, _ = sevenScenes.load_sample(source_1_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_2_rgb, _, source_2_cam, _ = sevenScenes.load_sample(source_2_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_3_rgb, _, source_3_cam, _ = sevenScenes.load_sample(source_3_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_4_rgb, _, source_4_cam, _ = sevenScenes.load_sample(source_4_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_5_rgb, _, source_5_cam, _ = sevenScenes.load_sample(source_5_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                    source_6_rgb, _, source_6_cam, _ = sevenScenes.load_sample(source_6_sample_path,
                                                                               cfg.dataset.image_height,
                                                                               cfg.dataset.image_width)

                except:
                    print("invalid_camera")
                    continue

                ref_rgb = ref_rgb.to(device)
                source_1_rgb = source_1_rgb.to(device)
                source_2_rgb = source_2_rgb.to(device)
                source_3_rgb = source_3_rgb.to(device)
                source_4_rgb = source_4_rgb.to(device)
                source_5_rgb = source_5_rgb.to(device)
                source_6_rgb = source_6_rgb.to(device)

                c, h, w = ref_rgb.shape

                gt_depth = gt_depth.to(device)  # [h, w]

                ref_cam = ref_cam.to(device)
                source_1_cam = source_1_cam.to(device)
                source_2_cam = source_2_cam.to(device)
                source_3_cam = source_3_cam.to(device)
                source_4_cam = source_4_cam.to(device)
                source_5_cam = source_5_cam.to(device)
                source_6_cam = source_6_cam.to(device)

                ref_rgb_batch = ref_rgb.unsqueeze(0).repeat(6, 1, 1, 1)
                source_rgb_batch = torch.cat([source_1_rgb.unsqueeze(0), source_2_rgb.unsqueeze(0),
                                              source_3_rgb.unsqueeze(0), source_4_rgb.unsqueeze(0),
                                              source_5_rgb.unsqueeze(0), source_6_rgb.unsqueeze(0)], dim=0)
                ref_cam_batch = torch.cat([ref_cam.unsqueeze(0), ref_cam.unsqueeze(0),
                                           ref_cam.unsqueeze(0), ref_cam.unsqueeze(0),
                                           ref_cam.unsqueeze(0), ref_cam.unsqueeze(0)], dim=0)
                source_cam_batch = torch.cat([source_1_cam.unsqueeze(0), source_2_cam.unsqueeze(0),
                                              source_3_cam.unsqueeze(0), source_4_cam.unsqueeze(0),
                                              source_5_cam.unsqueeze(0), source_6_cam.unsqueeze(0)], dim=0)

                start_time = time.monotonic()
                # idepth_preds_01, iconv_01 = depth_network(ref_rgb.unsqueeze(0), source_1_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_1_cam.unsqueeze(0))
                # idepth_preds_02, iconv_02 = depth_network(ref_rgb.unsqueeze(0), source_2_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_2_cam.unsqueeze(0))
                # idepth_preds_03, iconv_03 = depth_network(ref_rgb.unsqueeze(0), source_3_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_3_cam.unsqueeze(0))
                # idepth_preds_04, iconv_04 = depth_network(ref_rgb.unsqueeze(0), source_4_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_4_cam.unsqueeze(0))
                # idepth_preds_05, iconv_05 = depth_network(ref_rgb.unsqueeze(0), source_5_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_5_cam.unsqueeze(0))
                # idepth_preds_06, iconv_06 = depth_network(ref_rgb.unsqueeze(0), source_6_rgb.unsqueeze(0),
                #                                           ref_cam.unsqueeze(0), source_6_cam.unsqueeze(0))
                #
                #
                # idepth_refined, prob_map = depth_refine_network(idepth01=(idepth_preds_01[0] +
                #                                                           idepth_preds_03[0] + idepth_preds_05[0]) / 3.,
                #                                                 idepth02=(idepth_preds_02[0] +
                #                                                           idepth_preds_04[0] + idepth_preds_06[0]) / 3.,
                #                                                 iconv01=(iconv_01 + iconv_03 + iconv_05) / 3.,
                #                                                 iconv02=(iconv_02 + iconv_04 + iconv_06) / 3.)
                idepth_preds_01, iconv_01 = depth_network(ref_rgb_batch, source_rgb_batch,
                                                          ref_cam_batch, source_cam_batch)

                idepth_refined, prob_map = depth_refine_network(idepth01=(idepth_preds_01[0][0:1] +
                                                                          idepth_preds_01[0][2:3] + idepth_preds_01[0][
                                                                                                    4:5]) / 3.,
                                                                idepth02=(idepth_preds_01[0][1:2] +
                                                                          idepth_preds_01[0][3:4] + idepth_preds_01[0][
                                                                                                    5:6]) / 3.,
                                                                iconv01=(iconv_01[0:1] + iconv_01[2:3] + iconv_01[
                                                                                                         4:5]) / 3.,
                                                                iconv02=(iconv_01[1:2] + iconv_01[3:4] + iconv_01[
                                                                                                         5:6]) / 3.)
                end_time = time.monotonic()
                time_interval = (end_time - start_time)
                total_time += time_interval

                depth_preds = torch.div(1.0, idepth_refined.squeeze(1))
                intrinsic = ref_cam[1, 0:3, 0:3]
                intrinsic_inv = torch.inverse(intrinsic)
                normal_from_depth, _ = depth2normal(depth_preds, intrinsic_inv.unsqueeze(0))

                # ================================================================== #
                #                        Tensorboard Logging                         #
                # ================================================================== #
                with torch.no_grad():
                    info = {'rgb': ref_rgb.permute(1, 2, 0).cpu().numpy(),
                            'normal_from_depth': normal_from_depth.permute(0, 2, 3, 1).squeeze().cpu().numpy(),
                            'gt_depth':
                                gt_depth.cpu().numpy(),
                            'pred_idepth':
                                idepth_refined.squeeze().cpu().numpy(),
                            'prob_map': prob_map.squeeze().cpu().numpy()
                            }

                    rgb_filepath = os.path.join(rgb_dir, pred_depth_name.replace("pred_depth", "color"))
                    scipy.misc.imsave(rgb_filepath, info['rgb'])

                    pred_normal_filepath = os.path.join(pred_normal_dir,
                                                        pred_depth_name.replace("pred_depth.png", "pred_normal.npy"))
                    np.save(pred_normal_filepath, info['normal_from_depth'])

                    pred_normal_color = normal2color(info['normal_from_depth'])
                    pred_normal_color_filepath = os.path.join(pred_normal_dir,
                                                              pred_depth_name.replace(
                                                                  "pred_depth",
                                                                  "pred_normal"))
                    scipy.misc.imsave(pred_normal_color_filepath, pred_normal_color)

                    gt_depth_filepath = os.path.join(gt_depth_dir,
                                                     pred_depth_name.replace("pred_depth.png", "gt_depth.npy"))
                    np.save(gt_depth_filepath, info['gt_depth'])

                    gt_depth_color = depth2color(info['gt_depth'])
                    gt_depth_color_filepath = os.path.join(gt_depth_dir,
                                                           pred_depth_name.replace("pred_depth", "gt_depth"))
                    scipy.misc.imsave(gt_depth_color_filepath, gt_depth_color)

                    pred_depth = np.reciprocal(info['pred_idepth'] + 1e-4)
                    pred_depth[pred_depth > 100] = 0
                    pred_depth_filepath = os.path.join(pred_depth_dir,
                                                       pred_depth_name.replace("pred_depth.png", "pred_depth.npy"))
                    np.save(pred_depth_filepath, pred_depth)

                    pred_depth_color = depth2color(pred_depth)
                    pred_depth_color_filepath = os.path.join(pred_depth_dir, pred_depth_name)
                    scipy.misc.imsave(pred_depth_color_filepath, pred_depth_color)

                    prob_map_color = colorize_probmap(info['prob_map'])
                    prob_map_color_filepath = os.path.join(prob_map_dir,
                                                           pred_depth_name.replace("pred_depth.png", "prob_map.png"))
                    scipy.misc.imsave(prob_map_color_filepath, prob_map_color)

                    prob_map_filepath = os.path.join(prob_map_dir,
                                                     pred_depth_name.replace("pred_depth.png", "prob_map.npy"))
                    np.save(prob_map_filepath, info['prob_map'])


@ex.command
def cal_metrics(dataDir):
    GT_DATADIR = "/home/xiaoxiao/disk6/7_Scenes_dataset/"

    l1_errors = []
    abs_relative_errors = []
    sq_rel_errors = []
    rmse_log_errors = []
    rmse_errors = []
    scale_invariant_errors = []
    a1_errors = []
    a2_errors = []
    a3_errors = []

    MIN_DEPTH = 0.3
    MAX_DEPTH = 8.0

    for scene in sorted(os.listdir(dataDir)):
        if not os.path.isdir(os.path.join(dataDir, scene)):
            continue
        for seq in os.listdir(os.path.join(dataDir, scene)):
            if not seq.startswith('seq'):
                continue
            print(scene, seq)
            gt_depth_dir = os.path.join(dataDir, scene, seq, "gt_depth")
            pred_depth_dir = os.path.join(dataDir, scene, seq, "pred_depth")

            for filename in sorted(os.listdir(gt_depth_dir)):
                if filename.endswith(".npy"):
                    gt_depth = cv2.imread(
                        os.path.join(GT_DATADIR, scene, seq, filename.replace("gt_depth.npy", "depth.png")), -1) / 1000.

                    H, W = gt_depth.shape

                    pred_depth = np.load(os.path.join(pred_depth_dir, filename.replace("gt_depth", "pred_depth")))
                    pred_depth = cv2.resize(pred_depth, (W, H), cv2.INTER_LINEAR)
                    pred_depth[pred_depth < MIN_DEPTH] = MIN_DEPTH
                    pred_depth[pred_depth > MAX_DEPTH] = MAX_DEPTH
                    # pred_depth = pred_depth[0:480, 0:640]
                    valid_mask = compute_valid_depth_mask(gt_depth, min_thred=MIN_DEPTH, max_thred=MAX_DEPTH)
                    # pred_depth[pred_depth == 0] = 1e-5
                    gt_depth = gt_depth[valid_mask]
                    pred_depth = pred_depth[valid_mask]
                    l1_error = l1(gt_depth, pred_depth)
                    abs_relative_error = abs_relative(depth_gt=gt_depth, depth_pred=pred_depth)
                    rmse_error = rmse(gt_depth, pred_depth)
                    scale_invariant_error = scale_invariant(gt_depth, pred_depth)
                    sq_rel_error = sq_relative(pred_depth, gt_depth)
                    rmse_log_error = rmse_log(gt_depth, pred_depth)

                    a1 = ratio_threshold(gt_depth, pred_depth, 1.25)
                    a2 = ratio_threshold(gt_depth, pred_depth, 1.25 * 1.25)
                    a3 = ratio_threshold(gt_depth, pred_depth, 1.25 * 1.25 * 1.25)

                    l1_errors.append(l1_error)
                    abs_relative_errors.append(abs_relative_error)
                    rmse_errors.append(rmse_error)
                    sq_rel_errors.append(sq_rel_error)
                    rmse_log_errors.append(rmse_log_error)
                    scale_invariant_errors.append(scale_invariant_error)
                    a1_errors.append(a1)
                    a2_errors.append(a2)
                    a3_errors.append(a3)

    mean_l1_error = np.mean(np.array(l1_errors))
    mean_abs_relative_error = np.mean(np.array(abs_relative_errors))
    mean_sq_rel_error = np.mean(np.array(sq_rel_errors))
    mean_rmse_error = np.mean(np.array(rmse_errors))
    mean_rmse_log_error = np.mean(np.array(rmse_log_errors))
    mean_scale_invariant_error = np.mean(np.array(scale_invariant_errors))

    mean_a1_error = np.mean(np.array(a1_errors))
    mean_a2_error = np.mean(np.array(a2_errors))
    mean_a3_error = np.mean(np.array(a3_errors))

    print("mean_l1_error", mean_l1_error)
    print("a<1.25", mean_a1_error)
    print("a<1.25^2", mean_a2_error)
    print("a<1.25^3", mean_a3_error)
    print("abs.rel", mean_abs_relative_error)
    print("sq.rel", mean_sq_rel_error)
    print("rmse", mean_rmse_error)
    print("rmse_log", mean_rmse_log_error)
    print("scale.inv", mean_scale_invariant_error)

    file = open(os.path.join(dataDir, "evaluation_errors.txt"), 'w+')
    file.write("mean_l1_error: " + str(mean_l1_error) + "\n")
    file.write("a<1.25: " + str(mean_a1_error) + "\n")
    file.write("a<1.25^2: " + str(mean_a2_error) + "\n")
    file.write("a<1.25^3: " + str(mean_a3_error) + "\n")
    file.write("abs.rel: " + str(mean_abs_relative_error) + "\n")
    file.write("sq.rel: " + str(mean_sq_rel_error) + "\n")
    file.write("rmse: " + str(mean_rmse_error) + "\n")
    file.write("rmse log: " + str(mean_rmse_log_error) + "\n")
    file.write("scale.inv: " + str(mean_scale_invariant_error) + "\n")
    file.close()


if __name__ == "__main__":
    ex.add_config('./configs/config.yaml')
    # ex.run_commandline()
    cal_metrics("../evaluations_7_scenes_refine/5/")
