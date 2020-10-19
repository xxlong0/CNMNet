import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn as nn
import numpy as np
import cv2
import tensorflow as tf
from copy import deepcopy
from depthnet.inverse_warp import *
from scannet.preprocess import *


def get_pixel_coordinates(height, width):
    """based on the height and width, return pixel_coordinates"""
    pixel_coordinates = np.indices([width, height]).astype(np.float32)  # there should be [width, height]
    pixel_coordinates = np.concatenate((pixel_coordinates,
                                        np.ones([1, width, height])), axis=0)
    pixel_coordinates = np.reshape(pixel_coordinates, [3, -1])

    pixel_coordinates = torch.cuda.FloatTensor(pixel_coordinates)
    return pixel_coordinates


def process_camera_parameters(left_cam, right_cam, pixel_coordinates):
    """
    This function is used to process the two cameras' parameters, and get KRKiUV_T, KT_T
    :param left_cam: the reference camera, use corresponding depthmap to train
    :param right_cam: mapping from ref camera to source cam
    :return: K_{right}RK_{left}^{i}UV_T
    :return: K_{right}T_T
    """
    left_extrinsic = left_cam[:, 0, 0:4, 0:4]
    left_K = left_cam[:, 1, 0:3, 0:3]
    right_extrinsic = right_cam[:, 0, 0:4, 0:4]
    right_K = right_cam[:, 1, 0:3, 0:3]

    right2left = torch.matmul(right_extrinsic, left_extrinsic.inverse())

    right_in_left_T = right2left[:, 0:3, 3]
    right_in_left_R = right2left[:, 0:3, 0:3]

    KRK_i = torch.matmul(right_K, torch.matmul(right_in_left_R, left_K.inverse()))
    KRKiUV = torch.matmul(KRK_i, pixel_coordinates)

    ## because KT's shape is [3], so cannort use batch torch.matmul, use for loop
    KT = []
    for i in range(right_K.shape[0]):
        KT_1 = torch.matmul(right_K[i], right_in_left_T[i])
        KT.append(KT_1)
    KT = torch.stack(KT, dim=0)

    KT = KT.unsqueeze(-1)
    KT_cuda_Tensor = KT.type(torch.cuda.FloatTensor)
    KRKiUV_cuda_Tensor = KRKiUV.type(torch.cuda.FloatTensor)

    return KRKiUV_cuda_Tensor, KT_cuda_Tensor


def np2Depth(input_tensor, depth_scale=8.0):
    normalized = (input_tensor - 0.1) / depth_scale * 255.0
    normalized = normalized.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(normalized[i], cv2.COLORMAP_RAINBOW)
        return normalized_color
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)
        return normalized


def colorize_probmap(input_tensor):
    input_tensor = input_tensor * 255.0
    input_tensor = input_tensor.astype(np.uint8)
    if len(input_tensor.shape) == 3:
        normalized_color = np.zeros((input_tensor.shape[0],
                                     input_tensor.shape[1],
                                     input_tensor.shape[2],
                                     3))
        for i in range(input_tensor.shape[0]):
            normalized_color[i] = cv2.applyColorMap(input_tensor[i], cv2.COLORMAP_RAINBOW)
        return normalized_color
    if len(input_tensor.shape) == 2:
        normalized = cv2.applyColorMap(input_tensor, cv2.COLORMAP_RAINBOW)
        return normalized


def normal2color(normal_map):
    """
    colorize normal map
    :param normal_map: range(-1, 1)
    :return:
    """
    tmp = normal_map / 2. + 0.5  # mapping to (0, 1)
    color_normal = (tmp * 255).astype(np.uint8)

    return color_normal


def colorize_uvnormal(uvnormal):
    """

    :param uvnormal:
    :return:
    """
    b, c, h, w = uvnormal.size()
    ones = -torch.ones([b, 1, h, w]).type_as(uvnormal).to(uvnormal.device)

    normal = torch.cat([uvnormal, ones], dim=1)  # [b ,3, h, w]

    normal = normal / (torch.norm(normal, dim=1, keepdim=True) + 1e-5)

    color_normal = normal2color(normal.permute(0, 2, 3, 1).cpu().numpy())

    return color_normal



def depth2color(depth, MIN_DEPTH=0.3, MAX_depth=8.0):
    """
    colorize depth map
    :param depth:
    :return:
    """
    depth_clip = deepcopy(depth)
    depth_clip[depth_clip < MIN_DEPTH] = 0
    depth_clip[depth_clip > MAX_depth] = 0
    normalized = (depth_clip - MIN_DEPTH) / (MAX_depth - MIN_DEPTH) * 255.0
    normalized = [normalized, normalized, normalized]
    normalized = np.stack(normalized, axis=0)
    normalized = np.transpose(normalized, (1, 2, 0))
    normalized = normalized.astype(np.uint8)

    return cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)


class Depth2normal(nn.Module):
    def __init__(self, k_size=9):
        """
        convert depth map to point cloud first, and then calculate normal map
        :param k_size: the kernel size for neighbor points
        """
        super(Depth2normal, self).__init__()
        self.k_size = k_size

    def forward(self, depth, intrinsic_inv, instance_segs=None, planes_num=None):
        """

        :param depth: [B, H, W]
        :param intrinsic_inv: [B, 3, 3]
        :param instance_segs: [B, 20, h, w] stores "planes_num" plane instance seg (bool map)
        :param planes_num: [B]
        :return:
        """
        device = depth.get_device()
        b, h, w = depth.shape
        points = pixel2cam(depth, intrinsic_inv)  # (b, c, h, w)

        valid_condition = ((depth > 0) & (depth < 10.0)).type(torch.FloatTensor)
        valid_condition = valid_condition.unsqueeze(1)  # (b, 1, h, w)

        unford = torch.nn.Unfold(kernel_size=(self.k_size, self.k_size), padding=self.k_size // 2, stride=(1, 1))
        torch_patches = unford(points)  # (N,C×∏(kernel_size),L)
        matrix_a = torch_patches.view(-1, 3, self.k_size * self.k_size, h, w)
        matrix_a = matrix_a.permute(0, 3, 4, 2, 1)  # (b, h, w, self.k_size*self.k_size, 3)

        valid_condition = unford(valid_condition)
        valid_condition = valid_condition.view(-1, 1, self.k_size * self.k_size, h, w)
        valid_condition = valid_condition.permute(0, 3, 4, 2, 1)  # (b, h, w, self.k_size*self.k_size, 1)
        valid_condition_all = valid_condition.repeat([1, 1, 1, 1, 3])
        valid_condition_all = (valid_condition_all > 0.5).to(device)

        matrix_a_zero = torch.zeros_like(matrix_a)
        matrix_a_valid = torch.where(valid_condition_all, matrix_a, matrix_a_zero)
        matrix_a_trans = torch.transpose(matrix_a_valid, 3, 4).view(-1, 3, self.k_size * self.k_size).to(device)
        matrix_a_valid = matrix_a_valid.view(-1, self.k_size * self.k_size, 3).to(device)
        matrix_b = torch.ones([b, h, w, self.k_size * self.k_size, 1]).view([-1, self.k_size * self.k_size, 1]).to(
            device)

        point_multi = torch.bmm(matrix_a_trans, matrix_a_valid).to(device)

        matrix_det = point_multi.det()

        inverse_condition_invalid = torch.isnan(matrix_det) | (matrix_det < 1e-5)
        inverse_condition_valid = (inverse_condition_invalid == False)
        inverse_condition_valid = inverse_condition_valid.unsqueeze(1).unsqueeze(2)
        inverse_condition_all = inverse_condition_valid.repeat([1, 3, 3]).to(device)

        diag_constant = torch.ones([3])
        diag_element = torch.diag(diag_constant)
        diag_element = diag_element.unsqueeze(0).to(device)
        diag_matrix = diag_element.repeat([inverse_condition_all.shape[0], 1, 1])

        inversible_matrix = torch.where(inverse_condition_all, point_multi, diag_matrix)
        inv_matrix = torch.inverse(inversible_matrix)

        generated_norm = torch.matmul(torch.matmul(inv_matrix, matrix_a_trans), matrix_b).squeeze(-1)  # [-1, 3]
        generated_norm_normalize = generated_norm / (torch.norm(generated_norm, dim=1, keepdim=True) + 1e-5)

        generated_norm_normalize = generated_norm_normalize.view(b, h, w, 3).type(torch.FloatTensor).to(device)

        if planes_num is not None:

            instance_segs = instance_segs.unsqueeze(-1).repeat(1, 1, 1, 1, 3)  # [b, 20, h, w, 3]
            generated_norm_normalize_new = []

            loss = 0

            for b_i in range(len(planes_num)):

                generated_norm_normalize_bi = generated_norm_normalize[b_i:b_i + 1]
                zeros_tensor = torch.zeros_like(generated_norm_normalize_bi)
                # use plane segs to regularize the normal values
                for i in range(planes_num[b_i]):
                    instance_seg = instance_segs[b_i:b_i + 1, i, :, :, :]  # [1, h, w, 3]
                    nominator = torch.sum(
                        torch.mul(generated_norm_normalize_bi,
                                  instance_seg.type(torch.FloatTensor).to(device)).view(1, h * w, 3), dim=1)
                    denominator = torch.sum(
                        instance_seg[:, :, :, 0].view(1, h * w), dim=1, keepdim=True).type(torch.FloatTensor).to(device)

                    normal_regularized = (nominator / denominator).unsqueeze(1).unsqueeze(1).repeat(
                        1, h, w, 1)
                    normal_original = torch.where(instance_seg, generated_norm_normalize_bi, zeros_tensor)

                    similarity = torch.nn.functional.cosine_similarity(
                        normal_regularized.view(-1, 3), normal_original.view(-1, 3), dim=1)

                    loss += torch.mean(1 - similarity)

                    generated_norm_normalize_bi = torch.where(instance_seg, normal_regularized,
                                                              generated_norm_normalize_bi)
                generated_norm_normalize_new.append(generated_norm_normalize_bi)
            generated_norm_normalize = torch.stack(generated_norm_normalize_new, dim=0).squeeze(1)
            return generated_norm_normalize.permute(0, 3, 1, 2), loss, points

        return generated_norm_normalize.permute(0, 3, 1, 2), points


def get_normal_by_planes(gt_normal, instance_segs, planes_num):
    """
    use plane segmentation to refine the ground truth normal map from
    :param gt_normal: CUDA Tensor [b, 3, h, w]
    :param instance_segs: CUDA Tensor [B, 20, h, w] stores "planes_num" plane instance seg (bool map)
    :param planes_num:
    :return: refined normal map
    """
    device = gt_normal.get_device()
    instance_segs = instance_segs.unsqueeze(-1).repeat(1, 1, 1, 1, 3)  # [b, 20, h, w, 3]
    generated_norm_normalize_new = []

    b, c, h, w = gt_normal.shape
    normal_refined = gt_normal.clone()
    normal_refined = normal_refined.permute(0, 2, 3, 1)  # [b, h, w, 3]

    for b_i in range(len(planes_num)):

        generated_norm_normalize_bi = normal_refined[b_i:b_i + 1]
        # use plane segs to regularize the normal values
        for i in range(planes_num[b_i]):
            instance_seg = instance_segs[b_i:b_i + 1, i, :, :, :]  # [1, h, w, 3]
            nominator = torch.sum(
                torch.mul(generated_norm_normalize_bi,
                          instance_seg.type(torch.FloatTensor).to(device)).view(1, h * w, 3), dim=1)
            denominator = torch.sum(
                instance_seg[:, :, :, 0].view(1, h * w), dim=1, keepdim=True).type(torch.FloatTensor).to(device)

            normal_regularized = (nominator / denominator).unsqueeze(1).unsqueeze(1).repeat(
                1, h, w, 1)

            generated_norm_normalize_bi = torch.where(instance_seg, normal_regularized,
                                                      generated_norm_normalize_bi)
        generated_norm_normalize_new.append(generated_norm_normalize_bi)
    normal_refined = torch.stack(generated_norm_normalize_new, dim=0).squeeze(1)
    return normal_refined.permute(0, 3, 1, 2)
