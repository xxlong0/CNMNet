import os
from glob import glob
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_dir", type=str, required=True, help="where the dataset is stored")
parser.add_argument("--img_height", type=int, default=968, help="image height")
parser.add_argument("--img_width", type=int, default=1296, help="image width")
parser.add_argument("--depth_min", type=int, default=300, help="the minimum depth value")
parser.add_argument("--depth_interval", type=int, default=35, help="the depth interval of sweeping planes")
parser.add_argument("--img_height_original", type=int, default=968, help="image's original height")
parser.add_argument("--img_width_original", type=int, default=1296, help="image's original width")
args = parser.parse_args()

class camera_information_maker(object):
    def __init__(self,
                dataset_dir,
                img_width,
                img_height,
                depth_min,
                depth_interval,
                img_width_original,
                img_height_original):
        self.dataset_dir = dataset_dir
        self.img_width = img_width
        self.img_height = img_height
        self.img_width_original = img_width_original
        self.img_height_original = img_height_original
        self.depth_min = depth_min
        self.depth_interval = depth_interval
        self.train_seqs = sorted(os.listdir(self.dataset_dir))
        self.zoom_y = self.img_height/float(self.img_height_original)
        self.zoom_x = self.img_width/float(self.img_width_original)

    def scale_intrinsics(self, mat, sx, sy):
        out = np.copy(mat)
        out[0,0] *= sx
        out[0,2] *= sx
        out[1,1] *= sy
        out[1,2] *= sy
        return out

    def read_intrisic_file(self, filepath):
        """Read in the color_intrisic.txt and parse into a ndarray"""
        intrinsics = np.loadtxt(filepath, dtype='f', delimiter=' ')

        return intrinsics

    def read_pose_file(self, filepath):
        """Read in the pose/*.txt and parse into a ndarray"""
        pose = np.loadtxt(filepath, dtype='f', delimiter=' ')

        return pose

    def pose2extrinsics(self, pose):
        """Convert pose(camera to world) to extrinsic matrix(world to camera)"""
        extrinsics = np.linalg.inv(pose)
        return extrinsics

    def load_intrinsics(self, dataset_dir, drive):
        calib_file = os.path.join(dataset_dir, '%s/intrinsic/intrinsic_color.txt' % drive)
        intrinsics = self.read_intrisic_file(calib_file)
        intrinsics = intrinsics[:3, :3]
        return intrinsics

    def load_pose(self, pose_dir, id):
        pose_file = os.path.join(pose_dir, '%s.txt' %id)
        pose = self.read_pose_file(pose_file)
        return pose

    def write_cameras(self, intrinsics, pose, cameras_dir, id):
        filepath = os.path.join(cameras_dir, "%d_cam.txt" % id)
        file = open(filepath, "w")
        # file.write("#pose(camera to world, np.inverse(extrinsic matrix))\n")
        file.write("extrinsic\n")
        matrix = np.matrix(self.pose2extrinsics(pose))
        for line in matrix:
            np.savetxt(file, line, fmt='%.6f')
        file.write('\n')

        file.write("intrinsic \n")
        matrix = np.matrix(intrinsics)
        for line in matrix:
            np.savetxt(file, line, fmt='%.6f')
        file.write('\n')
        file.write(str(self.depth_min)+" "+str(self.depth_interval))
        file.close()

    def make_cameras(self):
        for seq in self.train_seqs:
            print(seq)
            seq_dir = os.path.join(self.dataset_dir, '%s' % seq)
            pose_dir = os.path.join(seq_dir, 'pose')
            camera_dir = os.path.join(seq_dir, 'cameras')

            if not os.path.exists(pose_dir):
                break

            if not os.path.exists(camera_dir):
                os.mkdir(camera_dir)

            intrinsics = self.load_intrinsics(self.dataset_dir, seq)
            intrinsics = self.scale_intrinsics(intrinsics, self.zoom_x, self.zoom_y)
            N = len(glob(pose_dir + '/*.txt'))
            for id in range(N):
                poses = self.load_pose(pose_dir, id)
                self.write_cameras(intrinsics, poses, camera_dir, id)

def main():
    maker = camera_information_maker(dataset_dir=args.dataset_dir,
                                    img_width=args.img_width,
                                    img_height=args.img_height,
                                    depth_min=args.depth_min,
                                    depth_interval=args.depth_interval,
                                    img_width_original=args.img_width_original,
                                    img_height_original=args.img_height_original)
    maker.make_cameras()

main()
