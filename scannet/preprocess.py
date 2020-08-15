from __future__ import print_function, division  # use python3's print and division
import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import random
import cv2
import glob
import re
from copy import copy
import shutil
from tensorflow.python.lib.io import file_io


def normalize_image(img):
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


def load_cam(file):
    """read camera txt file"""
    cam = np.zeros((2, 4, 4))
    words = file.read().split()

    # read extrinsic(world to camera) or pose(camera to world)
    for i in range(0, 4):
        for j in range(0, 4):
            extrinsic_index = 4 * i + j + 1
            cam[0][i][j] = words[extrinsic_index]

    # read intrinsic
    for i in range(0, 3):
        for j in range(0, 3):
            intrinsic_index = 3 * i + j + 18
            cam[1][i][j] = words[intrinsic_index]

    return cam


def load_label(file, labels_dict):
    """
    This function is used to load scannet segmentation label map.
    Because original labels in ScanNet are quite sparse and not successive,
    we use this function to map original labels to {0, 1, 2, ..., labels_num - 1}
    :param file: The file of label map
    :param labels_dict: the label mapping dictionary
    :return: processed label map, shape is [h, w]
    """
    label = cv2.imread(file, -1)  # the label map is saved in 16-bit

    # background and no mapping original labels are set to 0
    return np.vectorize(labels_dict.get)(label, 0.0)


def mask_depth_image(depth_image, min_depth, max_depth):
    """mask out-of-range pixel to zero"""

    # for cv2.threshold src is 8bit or 32 bit floating type
    depth_image = np.float32(depth_image)
    ret, depth_image = cv2.threshold(depth_image, min_depth, 100000, cv2.THRESH_TOZERO)
    ret, depth_image = cv2.threshold(depth_image, max_depth, 100000, cv2.THRESH_TOZERO_INV)
    depth_image = np.expand_dims(depth_image, 2)  # shape: (h, w, 1)
    depth_image[depth_image == 0.0] = 1e-6  # avoid zero depths
    return depth_image


def scale_camera(cam, scale_x, scale_y):
    """scale the camera intrinsics of one view"""
    new_cam = copy(cam)
    # focal
    new_cam[1][0][0] *= scale_x
    new_cam[1][1][1] *= scale_y

    # principle point:
    new_cam[1][0][2] *= scale_x
    new_cam[1][1][2] *= scale_y

    return new_cam


def scale_batch_cams(cams, scale_x, scale_y):
    """resize the camera intrinsics based on scale
    cams: shape [batch, 2, 4, 4]
    """
    for batch in range(cams.shape[0]):
        cams[batch] = scale_camera(cams[batch], scale_x, scale_y)

    return cams


def np2Img(np_image, Normalize=True):
    np_image = np.moveaxis(np_image, 0, -1)
    if Normalize:
        normalized = (np_image - np_image.min()) / (
                np_image.max() - np_image.min()) * 255.0
    else:
        normalized = np_image
    normalized = normalized[:, :, [2, 1, 0]]
    normalized = normalized.astype(np.uint8)
    return normalized


def np2Depth(input_tensor, minDepth=0.3, maxDepth=8.0):
    input_tensor = np.squeeze(input_tensor)
    normalized = (input_tensor - (1.0 / maxDepth)) / ((1.0 / minDepth) - (1.0 / maxDepth)) * 255.0
    normalized = normalized.astype(np.uint8)
    normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)

    return normalized


def np2Label(label, nclasses=10):
    label = np.squeeze(label)
    normalized = label / (nclasses + 0.0) * 255.0
    normalized = normalized.astype(np.uint8)
    normalized = cv2.applyColorMap(normalized, cv2.COLORMAP_RAINBOW)

    return normalized


def write_cam(file, pose, camera_k):
    # f = open(file, "w")
    f = open(file, "w+")

    f.write('extrinsic\n')
    for i in range(0, 4):
        for j in range(0, 4):
            f.write(str(pose[i][j]) + ' ')
        f.write('\n')
    f.write('\n')

    f.write('intrinsic\n')
    for i in range(0, 3):
        for j in range(0, 3):
            f.write(str(camera_k[i][j]) + ' ')
        f.write('\n')

    # f.write('\n' + str(cam[1][3][0]) + ' ' + str(cam[1][3][1]) + ' ' + str(cam[1][3][2]) + ' ' + str(cam[1][3][3]) + '\n')

    f.close()

def load_pfm(file):
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    header = file.readline().decode('latin-1').rstrip()

    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('latin-1'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')
    # scale = float(file.readline().decode('latin-1').rstrip())
    scale = float((file.readline().decode('latin-1')).rstrip())
    if scale < 0: # little-endian
        data_type = '<f'
    else:
        data_type = '>f' # big-endian
    data_string = file.read()
    data = np.fromstring(data_string, data_type)
    # data = np.fromfile(file, data_type)
    shape = (height, width, 3) if color else (height, width)
    data = np.reshape(data, shape)
    data = cv2.flip(data, 0)
    return data

def write_pfm(file, image, scale=1):
    file = file_io.FileIO(file, mode='wb')
    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n' if color else 'Pf\n')
    file.write('%d %d\n' % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n' % scale)

    image_string = image.tostring()
    file.write(image_string)

    file.close()
