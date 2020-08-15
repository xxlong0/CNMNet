from torch.utils.data import Dataset, DataLoader
from scannet.preprocess import *
import scipy.io
import os
import cv2
import numpy as np


class ScannetDataset(Dataset):
    def __init__(self, list_filepath, root_dir, view_num=3, interval=10, depth_scale=5.0, transform=None):

        self.list_filepath = list_filepath
        self.root_dir = root_dir
        self.view_num = view_num
        self.transform = transform
        self.interval = interval
        self.depth_scale = depth_scale
        self.sample_list = []

        self.load_sample_list()

    def load_sample_list(self):
        list_file = open(self.list_filepath, 'r')
        lines = list_file.readlines()
        list_file.close()

        self.sample_list = [line.split() for line in lines]

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        data = self.sample_list[index]  # scene_id, image_id
        # print(data[0], data[1])
        rgbs = []
        depths = []
        normals = []
        cameras = []

        filenames = []

        # reference view
        single_sample = self.load_single_sample(data[0], data[1])
        rgbs.append(single_sample['rgb'])
        depths.append(single_sample['depth'])
        normals.append(single_sample['normal'])
        cameras.append(single_sample['camera'])
        filenames.append(single_sample['filename'])

        # source views
        for view in range(self.view_num):
            i = view - self.view_num // 2
            if i == 0:
                continue
            image_id = int(data[1]) + self.interval * i
            image_id = str(image_id)
            single_sample = self.load_single_sample_less(data[0], image_id)
            rgbs.append(single_sample['rgb'])
            # depths.append(single_sample['depth'])
            # disparities.append(single_sample['disparity'])
            # normals.append(single_sample['normal'])
            cameras.append(single_sample['camera'])

            filenames.append(single_sample['filename'])

        rgbs = np.stack(rgbs, axis=0)
        depths = np.stack(depths, axis=0)
        normals = np.stack(normals, axis=0)
        cameras = np.stack(cameras, axis=0)

        sample = {'rgbs': rgbs,
                  'depths': depths,
                  'normals': normals,
                  'cameras': cameras,
                  'filenames': filenames}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_normal_png(self, normal_filepath):
        normal = cv2.imread(normal_filepath, -1)
        normal = cv2.cvtColor(normal, cv2.COLOR_BGR2RGB)
        normal = np.float32(normal)
        normal = (normal / 255. - 0.5) * 2

        return normal

    def load_single_sample(self, scene_id, image_id):
        rgb_filepath = os.path.join(os.path.join(scene_id, 'rgb'), image_id + '.jpg')
        rgb_filepath = os.path.join(self.root_dir, rgb_filepath)

        depth_filepath = os.path.join(os.path.join(scene_id, 'depth'), image_id + '.png')
        depth_filepath = os.path.join(self.root_dir, depth_filepath)

        normal_filepath = os.path.join(os.path.join(scene_id, 'normal_color'), image_id + '.png')
        normal_filepath = os.path.join(self.root_dir, normal_filepath)

        camera_filepath = os.path.join(os.path.join(scene_id, 'cameras'), image_id + '_cam.txt')
        camera_filepath = os.path.join(self.root_dir, camera_filepath)

        # load sample
        try:
            rgb = cv2.imread(rgb_filepath, -1)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = normalize_image(rgb)
        except:
            print("load image error", rgb_filepath)
            exit(1)

        try:
            depth = cv2.imread(depth_filepath, -1) / 1000.
            depth = np.float32(depth)
            depth[depth < 0.1] = 0
            depth[depth > self.depth_scale] = 0

            # disparity = np.reciprocal(depth + 1e-4)
            # disparity[disparity < 0.02] = 0
            # disparity[disparity > 3.0] = 0
            if not np.max(depth) > 0.0:
                exit(1)
        except:
            print("depth error", depth_filepath)
            exit(1)

        # normal = scipy.io.loadmat(normal_filepath)
        # normal = np.dstack((normal['nx'], normal['ny'], normal['nz']))

        normal = self.load_normal_png(normal_filepath)

        # normal may have some Nan values
        normal = np.where(np.isnan(normal), 0, normal)

        camera = load_cam(open(camera_filepath, 'r'))
        camera = np.float32(camera)

        sample = {'rgb': rgb,
                  'depth': depth,
                  'normal': normal,
                  'camera': camera,
                  'filename': str(scene_id) + "_" + str(image_id)}

        return sample

    def load_single_sample_less(self, scene_id, image_id):
        rgb_filepath = os.path.join(os.path.join(scene_id, 'rgb'), image_id + '.jpg')
        rgb_filepath = os.path.join(self.root_dir, rgb_filepath)

        camera_filepath = os.path.join(os.path.join(scene_id, 'cameras'), image_id + '_cam.txt')
        camera_filepath = os.path.join(self.root_dir, camera_filepath)

        # load sample
        try:
            rgb = cv2.imread(rgb_filepath, -1)
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb = normalize_image(rgb)
        except:
            print("opencv error", rgb_filepath)
            exit(1)

        camera = load_cam(open(camera_filepath, 'r'))
        camera = np.float32(camera)

        sample = {'rgb': rgb,
                  'camera': camera,
                  'filename': str(scene_id) + "_" + str(image_id)}

        return sample

    def load_seg(self, plane_seg_filepath):
        seg = cv2.imread(plane_seg_filepath, -1)
        seg[seg == np.max(seg)] = 20  # non-planar set to 20

        return seg

    def load_plane_instance_seg(self, seg, plane_num):
        """
        This function will generate each plane's segmentation 0-1 map,
        should run after self.load_seg()
        :param seg:
        :return:
        """

        instance = np.zeros([20, seg.shape[0], seg.shape[1]], dtype=np.uint8)
        for i in range(plane_num):
            instance[i] = seg == i
            if np.sum(seg == i) < 100:
                raise Exception('wrong plane instance')

        return instance

    def process_by_seg(self, plane_para, plane_seg, scene_id, image_id):
        """
        plane_seg remove some small planes, plane_para and plane_error doesn't process
        :param plane_para:
        :param plane_error:
        :param plane_seg:
        :return:
        """
        plane_para_new = []
        i = 0
        if len(np.unique(plane_seg)) == 1:
            raise Exception('there are no planes in plane_seg', scene_id, image_id)
        for segmentIndex in np.unique(plane_seg):
            if segmentIndex == 20:
                continue
            plane_para_new.append(plane_para[segmentIndex])
            plane_seg[plane_seg == segmentIndex] = i
            i = i + 1

        plane_para_new = np.stack(plane_para_new, axis=0)
        return plane_para_new, plane_seg

    def plane_para_coordinate_exchange(self, plane_para):
        """
        the plane_para generated by planercnn has different coordinate
        with normal map generated from depth map
        :param plane_para:
        :return:
        """
        tmp = plane_para[:, 1].copy()
        plane_para[:, 1] = -plane_para[:, 2]
        plane_para[:, 2] = tmp

        return plane_para

    def normal_from_plane_para(self, plane_para, plane_num, seg):

        normal = np.zeros((seg.shape[0], seg.shape[1], 3))
        for i in range(plane_num):
            normal[seg == i] = plane_para[i]

        normal /= (np.linalg.norm(normal, ord=2, axis=2, keepdims=True) + 1e-5)

        return normal


class Resizer(object):

    def __init__(self, image_width_expected=1280, image_height_expected=960,
                 depth_width_expected=320, depth_height_expected=240):
        """
        :param self.image_width_expected:
        :param self.image_height_expected:
        :param self.depth_width_expected:
        :param self.depth_height_expected:
        """
        self.image_width_expected = image_width_expected
        self.image_height_expected = image_height_expected
        self.depth_width_expected = depth_width_expected
        self.depth_height_expected = depth_height_expected

    def __call__(self, sample):
        """
        :param sample: what the dataloader returns
        :return:
        """
        rgbs = sample['rgbs']
        depths = sample['depths']
        normals = sample['normals']
        cameras = sample['cameras']

        original_image_width = rgbs[0].shape[1]
        original_image_height = rgbs[0].shape[0]
        scale_x = float(self.image_width_expected) / original_image_width
        scale_y = float(self.image_height_expected) / original_image_height

        rgbs = self.scale_imgs(rgbs,
                               expected_height=self.image_height_expected,
                               expected_width=self.image_width_expected,
                               interpolation='linear')
        depths = self.scale_depths(depths,
                                   expected_height=self.depth_height_expected,
                                   expected_width=self.depth_width_expected,
                                   interpolation='nearest')
        normals = self.scale_imgs(normals,
                                  expected_height=self.depth_height_expected,
                                  expected_width=self.depth_width_expected,
                                  interpolation='nearest')
        cameras = self.scale_cams(cameras,
                                  scale_x=scale_x,
                                  scale_y=scale_y)

        sample = {'rgbs': rgbs,
                  'depths': depths,
                  'normals': normals,
                  'cameras': cameras,
                  'filenames': sample['filenames']}

        return sample

    def scale_img(self, image, expected_height, expected_width, interpolation):
        # although opencv load image in shape (height, width, channel), cv2.resize still need shape (width, height)
        if interpolation == 'linear':
            return cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_LINEAR)
        if interpolation == 'nearest':
            return cv2.resize(image, (expected_width, expected_height), interpolation=cv2.INTER_NEAREST)
        if interpolation is None:
            raise Exception('interpolation cannot be None')

    def scale_imgs(self, images, expected_height, expected_width, interpolation):
        images_new = np.zeros((images.shape[0], expected_height, expected_width, images.shape[3]))
        for view in range(images.shape[0]):
            images_new[view] = self.scale_img(images[view], expected_height, expected_width, interpolation)

        return images_new

    def scale_depths(self, depths, expected_height, expected_width, interpolation='nearest'):
        depths_new = np.zeros((depths.shape[0], expected_height, expected_width))
        for view in range(depths.shape[0]):
            depths_new[view] = self.scale_img(depths[view], expected_height, expected_width, interpolation)

        return depths_new

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

    def scale_cams(self, cams, scale_x, scale_y):
        for view in range(cams.shape[0]):
            cams[view] = self.scale_cam(cams[view], scale_x, scale_y)

        return cams

    def scale_instance_seg(self, instance_seg, expected_height, expected_width):
        instance_seg_new = np.zeros([instance_seg.shape[0], expected_height, expected_width], dtype=np.uint8)
        for i in range(instance_seg.shape[0]):
            instance_seg_new[i] = self.scale_img(instance_seg[i], expected_height, expected_width, 'nearest')

        return instance_seg_new

    def scale_instance_segs(self, instance_segs, expected_height, expected_width):
        instance_segs_new = np.zeros((instance_segs.shape[0], instance_segs.shape[1], expected_height, expected_width),
                                     dtype=np.uint8)
        for view in range(instance_segs.shape[0]):
            instance_segs_new[view] = self.scale_instance_seg(instance_segs[view], expected_height, expected_width)
        return instance_segs_new


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        rgbs = sample['rgbs']
        depths = sample['depths']
        normals = sample['normals']
        cameras = sample['cameras']

        # numpy to tensor
        rgbs = self.rgbs2tensor(rgbs)
        depths = self.depths2tensor(depths)
        depths = depths.unsqueeze(1)

        normals = self.rgbs2tensor(normals)
        cameras = self.cams2tensor(cameras)

        # disparity = np.reciprocal(depth + 1e-4)
        # disparity[disparity < 0.02] = 0
        # disparity[disparity > 3.0] = 0

        sample = {'rgbs': rgbs,
                  'depths': depths,
                  # 'disparities': disparities,
                  'normals': normals,
                  'cameras': cameras,
                  'filenames': sample['filenames']}

        return sample

    def rgbs2tensor(self, rgbs):
        """
        3-channel images
        :param rgbs:
        :return:
        """
        # swap color axis because
        # numpy image: views x H x W x C
        # torch image: views x C X H X W
        rgbs = np.float32(rgbs)
        rgbs = rgbs.transpose((0, 3, 1, 2))

        return torch.from_numpy(rgbs)

    def depths2tensor(self, depths):
        """
        2-channels images
        :param depths:
        :return:
        """
        return torch.from_numpy(np.float32(depths))

    def plane_segs2tensor(self, plane_segs):
        return torch.ByteTensor(plane_segs)

    def cams2tensor(self, cameras):
        return torch.from_numpy(cameras)

    def plane_paras2tensor(self, plane_paras):
        """
        because each rgb has different number of planes, plane_para has different size
        return list
        :param plane_paras:
        :return:
        """
        return [torch.Tensor(plane_para) for plane_para in plane_paras]

    def plane_errors2tensor(self, plane_errors):
        plane_errors_tensor = []
        for plane_error in plane_errors:
            plane_error = plane_error[()]  # recover dict from 0-size array
            # the key "plane_errors" is a list of planes' error
            plane_error_tensor = {'error': plane_error['error'],
                                  'plane_errors': torch.Tensor(plane_error['plane_errors'])}
            plane_errors_tensor.append(plane_error_tensor)

        return plane_errors_tensor

    def plane_instance_segs2tensor(self, plane_instance_segs):
        return torch.ByteTensor(plane_instance_segs)


if __name__ == "__main__":
    from torchvision import transforms, utils

    dataset = ScannetDataset(list_filepath='./train_plane_view3_scans0_59_interval2.txt',
                             root_dir='/home/xiaoxiao/disk2/ScanNet/rawData/scans',
                             transform=transforms.Compose(
                                 [Resizer(), ToTensor()]
                             ))
    dataloader = DataLoader(dataset=dataset, batch_size=2,
                            # todo: because return different size of tensors, batch_size can only be 1
                            shuffle=True, num_workers=4)

    # dataloader require tensor, if transform is None, returned value is not tensor
    for iter_num, data in enumerate(dataloader):
        print(iter_num)
        sample = data
