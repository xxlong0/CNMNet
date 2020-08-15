import os
import argparse
from glob import glob
import numpy as np
import cv2
import scipy.io

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--view_num", type=int, default=3, help="the number of views")
parser.add_argument("--error_thred", type=float, default=0.7, help="the error of generated depth with ground truth")
parser.add_argument("--sequence_interval", type=int, default=2, help="sequence_interval")
parser.add_argument("--scans_num", type=int, default=1, help="the number of scans")
parser.add_argument("--scans_start_id", type=int, default=24, help="the first selected scan's id")
parser.add_argument("--list_savepath", type=str, default="./", help="the directory which list generated is saved in")
parser.add_argument("--mode", type=str, default="train", help="generate train/test/validation data list")
args = parser.parse_args()

class scannet_list_maker(object):
    """generate data paths for scannet dataset
    """
    def __init__(self,
                dataset_dir,
                view_num=3,
                sequence_interval=10,
                list_savepath='./',
                mode='train'):
        self.dataset_dir = dataset_dir
        self.first_scan_id = args.scans_start_id
        self.last_scan_id = args.scans_start_id + args.scans_num
        self.data_seqs = sorted(os.listdir(self.dataset_dir))[self.first_scan_id:self.last_scan_id]
        self.data_list = []
        self.view_num = view_num
        self.list_savepath = list_savepath
        self.mode = mode
        self.sequence_interval = sequence_interval

    def is_valid(self,
                 dataset_dir,
                 camera_filepath,
                 plane_seg_filepath,
                 plane_filepath,
                 plane_error_filepath,
                 rgb_filepath,
                 depth_filepath,
                 normal_filepath):
        """In the original ScanNet dataset, some poses of frames is invalid which is
        a matrix whose elememts are all "-inf";
        And some rgbs have no planes, also invalid
        """
        try:
            # plane_error > 0.1 invalid
            plane_error_filepath = os.path.join(dataset_dir, plane_error_filepath)
            plane_error = np.load(plane_error_filepath, allow_pickle=True)[()]
            error = plane_error['error']
            if error > args.error_thred:
                print('plane_error > ', args.error_thred)
                return False
        except:
            print("error file wrong", plane_error_filepath)
            return False

        try:
            rgb = cv2.imread(os.path.join(dataset_dir, rgb_filepath), -1)
            if len(rgb.shape) != 3:
                print("error rgb", rgb_filepath)
                return False
        except:
            print("error", rgb_filepath)
            return False

        try:
            depth = cv2.imread(os.path.join(dataset_dir, depth_filepath), -1)
            if len(depth.shape) != 2:
                print("error depth", depth_filepath)
                return False
        except:
            print("error", depth_filepath)
            return False

        try:
            normal = scipy.io.loadmat(os.path.join(dataset_dir, normal_filepath))
            map_nx = np.invert(np.isnan(normal['nx']))
            if not np.all(map_nx):
                return False
            map_ny = np.invert(np.isnan(normal['ny']))
            if not np.all(map_ny):
                return False
            map_nz = np.invert(np.isnan(normal['nz']))
            if not np.all(map_nz):
                return False
        except:
            print("error", os.path.join(dataset_dir, normal_filepath))
            return False

        try:
            camera_filepath = os.path.join(dataset_dir, camera_filepath)
            cam = self.load_cam(open(camera_filepath))

            # camera pose is invalid
            for i in range(4):
                for j in range(4):
                    if np.isinf(cam[0][i][j]):
                        print("camera invalid", camera_filepath)
                        return False

            # there is no planes in this rgb
            plane_seg_filepath = os.path.join(dataset_dir, plane_seg_filepath)
            plane_seg = cv2.imread(plane_seg_filepath, -1)
            if len(np.unique(plane_seg)) == 1:
                print('no planes in plane_seg')
                return False

            plane_filepath = os.path.join(dataset_dir, plane_filepath)
            planes = np.load(plane_filepath)
            if len(planes) == 0:
                print('plane_para is None')
                return False
        except:
            return False

        return True

    def gen_scannet_list(self, mode='train'):
        for seq in self.data_seqs:
            seq_dir = os.path.join(self.dataset_dir, '%s' % seq)
            normal_dir = os.path.join(seq_dir, 'normal')
            camera_dir = os.path.join(seq_dir, 'cameras')
            plane_dir = os.path.join(seq_dir, 'planercnn_para_003')
            plane_error_dir = os.path.join(seq_dir, 'planercnn_error_003')
            N = len(glob(normal_dir + '/*.mat'))
            M = len(glob(camera_dir + '/*.txt'))
            if N > 0.8 * M:
                N = N // 5
            for i in range(int(self.view_num//2)*self.sequence_interval, N - int(self.view_num//2)*self.sequence_interval , self.sequence_interval):
                # reference view, save relative path
                plane_filepath = os.path.join('%s/planercnn_para_003' % seq, '%d_planes.npy' % int(i*5))
                plane_seg_filepath = os.path.join('%s/planercnn_seg_003' % seq, '%d.png' % int(i*5))
                camera_filepath = os.path.join('%s/cameras' % seq, '%d_cam.txt' % int(i*5))
                plane_error_filepath = os.path.join('%s/planercnn_error_003' % seq, '%d.npy' % int(i*5))
                rgb_filepath = os.path.join('%s/rgb' % seq, '%d.jpg' % int(i*5))
                depth_filepath = os.path.join('%s/depth' % seq, '%d.png' % int(i * 5))
                normal_filepath = os.path.join('%s/normal' % seq, '%d.mat' % int(i * 5))

                tmp_data_list=[]
                IS_VALID = True

                IS_VALID = IS_VALID and self.is_valid(self.dataset_dir, camera_filepath,
                                                      plane_seg_filepath, plane_filepath,
                                                      plane_error_filepath, rgb_filepath,
                                                      depth_filepath, normal_filepath)
                tmp_data_list.append('%s %d' % (seq, int(i*5)))
                for j in range(self.view_num):
                    if j == int(self.view_num//2):
                        continue
                    j_id = i + self.sequence_interval * (j - int(self.view_num//2))
                    plane_filepath = os.path.join('%s/planercnn_para_003' % seq, '%d_planes.npy' % int(j_id*5))
                    plane_seg_filepath = os.path.join('%s/planercnn_seg_003' % seq, '%d.png' % int(j_id * 5))
                    camera_filepath = os.path.join('%s/cameras' % seq, '%d_cam.txt' % int(j_id*5))
                    plane_error_filepath = os.path.join('%s/planercnn_error_003' % seq, '%d.npy' % int(j_id*5))
                    rgb_filepath = os.path.join('%s/rgb' % seq, '%d.jpg' % int(j_id*5))
                    depth_filepath = os.path.join('%s/depth' % seq, '%d.png' % int(j_id*5))
                    normal_filepath = os.path.join('%s/normal' % seq, '%d.mat' % int(j_id*5))
                    IS_VALID = IS_VALID and self.is_valid(self.dataset_dir, camera_filepath,
                                                          plane_seg_filepath, plane_filepath,
                                                          plane_error_filepath, rgb_filepath,
                                                          depth_filepath, normal_filepath)

                if IS_VALID:
                    self.data_list = self.data_list + tmp_data_list
                else:
                    print('invalid: ', camera_filepath)

    def load_cam(self, file, interval_scale=1):
        """ read camera txt file """
        cam = np.zeros((2, 4, 4))
        words = file.read().split()
        # read extrinsic
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

    def write_list(self, mode='train'):
        if mode == 'train':
            file = open(os.path.join(self.list_savepath, 'train_plane_view%d_scans%d_%d_interval%d_error%d.txt' % (self.view_num,  self.first_scan_id, self.last_scan_id-1, self.sequence_interval, args.error_thred)), 'w+')
        elif mode == 'validation':
            file = open(os.path.join(self.list_savepath, 'validation_plane_view%d_scans%d_%d_interval%d_error%d.txt' % (self.view_num,  self.first_scan_id, self.last_scan_id-1, self.sequence_interval, int(10*args.error_thred))), 'w+')
        for path in self.data_list:
            file.write(path+'\n')
        file.close()

    def generate_txt(self):
        self.gen_scannet_list(self.mode)
        self.write_list(self.mode)


def main():
    maker = scannet_list_maker(args.dataset_dir,
                               args.view_num,
                               args.sequence_interval,
                               args.list_savepath,
                               args.mode)
    maker.generate_txt()
    # maker.make_test_scan(0, './test_scan')

main()
