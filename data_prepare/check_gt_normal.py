import os
import cv2
import torch
import numpy as np

from depthnet.depth_util import Depth2normal
from depthnet.losses import surface_normal_loss

if __name__ == "__main__":
    gt_depth_filename = "/mnt/scannet/dps-train/sun3d_train_0.01m_to_0.1m_00000/0000.npy"
    gt_normal_filename = "/mnt/scannet/dps-train/sun3d_train_0.01m_to_0.1m_00000/normal/0000.npy"
    cam_filename = "/mnt/scannet/dps-train/sun3d_train_0.01m_to_0.1m_00000/cam.txt"

    gt_depth = np.load(gt_depth_filename)
    gt_normal = np.load(gt_normal_filename)
    intrinsics = np.loadtxt(cam_filename)

    gt_depth = torch.Tensor(gt_depth).unsqueeze(0).type(torch.FloatTensor).cuda()
    gt_normal = torch.Tensor(gt_normal).unsqueeze(0).type(torch.FloatTensor).cuda()
    intrinsics = torch.Tensor(intrinsics).unsqueeze(0).type(torch.FloatTensor).cuda()

    depth2normal = Depth2normal().cuda()

    est_normal, points = depth2normal(gt_depth, intrinsics.inverse())

    mask = (gt_depth > 0) & (~torch.isnan(gt_depth))

    est_normal_copy = est_normal.clone().permute(0, 2, 3, 1)
    est = est_normal_copy[0, 250, 200]
    gt = gt_normal[0, 250, 200]
    loss, mean_angle = surface_normal_loss(est_normal, gt_normal.permute(0, 3, 1, 2), mask.unsqueeze(1))

    print(mean_angle)
