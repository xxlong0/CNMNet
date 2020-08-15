import os
import numpy as np
from scannet.preprocess import *
import cv2
from pyntcloud import PyntCloud
import pandas as pd
import copy


class ColorPalette:
    def __init__(self, numColors):
        np.random.seed(2)
        self.colorMap = np.array([[255, 0, 0],
                                  [0, 255, 0],
                                  [0, 0, 255],
                                  [80, 128, 255],
                                  [255, 230, 180],
                                  [255, 0, 255],
                                  [0, 255, 255],
                                  [100, 0, 0],
                                  [0, 100, 0],
                                  [255, 255, 0],
                                  [50, 150, 0],
                                  [200, 255, 255],
                                  [255, 200, 255],
                                  [128, 128, 80],
                                  [0, 50, 128],
                                  [0, 100, 100],
                                  [0, 255, 128],
                                  [0, 128, 255],
                                  [255, 0, 128],
                                  [128, 0, 255],
                                  [255, 128, 0],
                                  [128, 255, 0],
        ])

        if numColors > self.colorMap.shape[0]:
            self.colorMap = np.concatenate([self.colorMap, np.random.randint(255, size = (numColors - self.colorMap.shape[0], 3))], axis=0)
            pass

        return

    def getColorMap(self):
        return self.colorMap

    def getColor(self, index):
        if index >= self.colorMap.shape[0]:
            return np.random.randint(255, size = (3))
        else:
            return self.colorMap[index]
            pass


def planes_label2color(plane_label, numColors):
    """
    colorize planes_label
    :param plane_label: [h, w]
    :return: [h, w, 3]
    """
    h, w = plane_label.shape
    color_palette = ColorPalette(numColors)
    colorMap = color_palette.getColorMap()
    colorMap[numColors-1] = [0, 0, 0]
    plane_label = np.reshape(plane_label, (h*w))

    color_label = np.asarray([colorMap[i] for i in plane_label])
    max = np.max(plane_label)
    color_label = np.reshape(color_label, (h, w, 3))
    return color_label



def read_plane_paramters_file(filepath):
    """
    read plane parameters from txt file:
    #plane_index number_of_points_on_the_plane
    plane_color_in_png_image(1x3)
    plane_normal(1x3)
    plane_center(1x3) sx sy sz sxx syy szz sxy syz sxz
    :param file:
    :return:
    """
    file = open(filepath, 'r')
    lines = file.readlines()
    planes = []
    for line in lines:
        if not line.startswith('#'):
            paras = line.split()
            plane = {'index': int(paras[0]),
                     'num_of_points': int(paras[1]),
                     'ratio': float(paras[1]) / (640. * 480.),
                     'nx': float(paras[5]),
                     'ny': float(paras[6]),
                     'nz': float(paras[7]),
                     'sx': float(paras[8]),
                     'sy': float(paras[9]),
                     'sz': float(paras[10])}

            normal = np.asarray([plane['nx'], plane['ny'], plane['nz']])
            center_point = np.asarray([plane['sx'], plane['sy'], plane['sz']]).transpose()
            offset = np.matmul(normal, 4.0 * center_point)
            plane['normal'] = normal
            plane['offset'] = offset

            planes.append(plane)

    return planes


def read_plane_label_file(filepath):
    """
    read txt file, plane labels map
    The index "num_of_planes" is non-planar region, not zero
    :param filepath:
    :return: [h, w]
    """
    file = open(filepath, 'r')
    lines = file.readlines()
    label = []
    for line in lines[1:]:
        label_line = line.split()  # width
        label.append([int(i) for i in label_line])

    label = np.asarray(label)
    return label


def uv2pointcloud(depth, valid, intrinsics):
    """
    use depth map and camera intrinsic to get point cloud
    :param depth:
    :param validmask:
    :param camera_K:
    :return:
    """

    fx = intrinsics[0][0]
    fy = intrinsics[1][1]
    cx = intrinsics[0][2]
    cy = intrinsics[1][2]

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
    z = np.where(valid, depth, np.nan)
    x = np.where(valid, z * (c - cx) / fx, 0)
    y = np.where(valid, z * (r - cy) / fy, 0)

    points = np.dstack((x, y, z))

    return points


def points_save_as_ply(points, rgb, savefilepath):
    """

    :param points:
    :param rgb:
    :param savefilepath:
    :return:
    """
    rows, cols, _ = points.shape
    data_list = []
    for i in range(rows):
        for j in range(cols):
            if np.isnan(points[i, j][2]):
                continue
            # make sure the DataFrame has the colours as uint8
            data = [points[i, j][0], points[i, j][1], points[i, j][2],
                    rgb[i, j][0], rgb[i, j][1], rgb[i, j][2]]
            # print(data)
            data_list.append(data)

    data_list = np.array(data_list)
    cloud = PyntCloud(pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=data_list,
        columns=["x", "y", "z", "red", "green", "blue"]))

    cloud.points["red"] = cloud.points["red"].astype(np.uint8)
    cloud.points["green"] = cloud.points["green"].astype(np.uint8)
    cloud.points["blue"] = cloud.points["blue"].astype(np.uint8)

    cloud.to_file(savefilepath)


def points2offset(points, normal, valid):
    """
    get plane's offset: nx=d
    d = sum(nx)/valid_num
    :param points: [h, w, 3]
    :param normal: [3, 1]
    :param valid:
    :return:
    """
    offset_map = np.matmul(points, normal)
    # offset_map = np.squeeze(offset_map, axis=2)
    offset_map = np.where(valid, offset_map, 0)
    offset = np.sum(offset_map) / float(np.sum(valid))

    return offset


def test1():
    plane_para_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0000_00/plane_peac_scale4/0.jpg-plane-data.txt"
    plane_label_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0000_00/plane_peac_scale4/0.jpg-plane-label.txt"
    camera_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0000_00/cameras/0_cam.txt"
    depth_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0000_00/depth/0.png"
    rgb_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0000_00/rgb/0.jpg"

    cam = load_cam(open(camera_filepath, 'r'))
    cam = scale_camera(cam, scale_x=0.5, scale_y=0.5)
    plane_para = read_plane_paramters_file(plane_para_filepath)
    plane_label = read_plane_label_file(plane_label_filepath)
    depth = cv2.imread(depth_filepath, -1)
    rgb = cv2.imread(rgb_filepath)
    rgb = cv2.resize(rgb, (640, 480))

    # cv2.imshow('plane_label', plane_label*1000)
    # cv2.waitKey(0)

    intrinsics = cam[1]
    label_index = 4
    points_1 = uv2pointcloud(depth, (plane_label == label_index), intrinsics)
    # points_save_as_ply(points_1, rgb, "points1.ply")
    offset_1 = points2offset(points_1, plane_para[label_index]['normal'], (plane_label == label_index))

    error = offset_1 - plane_para[label_index]['offset']
    print(error)


def angle_of_2vectors(v1, v2, acute):
    """
    calculate angle of v1 and v2
    :param v1: [n]
    :param v2: [n]
    :return:
    """
    angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
    if acute:
        return 180.0 * angle / np.pi
    else:
        return 180.0 * (2 * np.pi - angle) / np.pi


def merge_planes_in_one_seg_label(planes_para, planes_label, seg):
    """
    merge planes in one segment label map
    :param planes_para: 
    :param planes_label: 
    :param seg: segment map for only one label
    :return: 
    """
    while True:
        # merge planes in seg
        planes_num = len(planes_para)
        print("planes_num", planes_num)
        planes_num_before = copy.deepcopy(planes_num)
        planes_index_in_seg = []
        for i in range(planes_num):
            label_i = (planes_label == i)
            label_i_in_seg = np.logical_and(seg, label_i)
            print(np.sum(label_i_in_seg) / float(np.sum(label_i)))
            if np.sum(label_i_in_seg) / float(np.sum(label_i)) > 0.6:
                planes_index_in_seg.append(i)

        if len(planes_index_in_seg) >= 2:
            planes_in_seg_num = len(planes_index_in_seg)
            flag = False
            for i in range(planes_in_seg_num):
                for j in range(i + 1, planes_in_seg_num):
                    plane1_para = planes_para[planes_index_in_seg[i]]
                    plane2_para = planes_para[planes_index_in_seg[j]]
                    normal_1 = plane1_para['normal']
                    normal_2 = plane2_para['normal']
                    if angle_of_2vectors(normal_1, normal_2,
                                         True) < 5:  # if the angle of two normals < 5 degree, then merge
                        normal_new = plane1_para['ratio'] / (plane1_para['ratio'] + plane2_para['ratio']) * normal_1 + \
                                     plane2_para['ratio'] / (plane1_para['ratio'] + plane2_para['ratio']) * normal_2
                        sx_new = (plane1_para['num_of_points'] * plane1_para['sx']
                                  + plane2_para['num_of_points'] * plane2_para['sx']) \
                                 / (plane1_para['num_of_points'] + plane2_para['num_of_points'])
                        sy_new = (plane1_para['num_of_points'] * plane1_para['sy']
                                  + plane2_para['num_of_points'] * plane2_para['sy']) \
                                 / (plane1_para['num_of_points'] + plane2_para['num_of_points'])
                        sz_new = (plane1_para['num_of_points'] * plane1_para['sz']
                                  + plane2_para['num_of_points'] * plane2_para['sz']) \
                                 / (plane1_para['num_of_points'] + plane2_para['num_of_points'])

                        planes_para[planes_index_in_seg[i]]['num_of_points'] = (
                                    plane1_para['num_of_points'] + plane2_para['num_of_points'])
                        planes_para[planes_index_in_seg[i]]['ratio'] = plane1_para['ratio'] + plane2_para['ratio']
                        planes_para[planes_index_in_seg[i]]['nx'] = normal_new[0]
                        planes_para[planes_index_in_seg[i]]['ny'] = normal_new[1]
                        planes_para[planes_index_in_seg[i]]['nz'] = normal_new[2]

                        planes_para[planes_index_in_seg[i]]['sx'] = sx_new
                        planes_para[planes_index_in_seg[i]]['sy'] = sy_new
                        planes_para[planes_index_in_seg[i]]['sz'] = sz_new

                        planes_para[planes_index_in_seg[i]]['normal'] = normal_new

                        # change plane_label map
                        planes_label = np.where((planes_label == planes_index_in_seg[j]), planes_index_in_seg[i], planes_label)

                        for w in range(planes_index_in_seg[j] + 1, planes_num):
                            # print(w)
                            planes_para[w]['index'] -= 1
                            planes_label = np.where((planes_label == w), w - 1,
                                                    planes_label)

                        planes_label = np.where((planes_label == planes_num), planes_num - 1,
                                                planes_label)

                        del planes_para[planes_index_in_seg[j]]
                        flag = True
                        break

                if flag:
                    break
            if planes_num_before == len(planes_para):
                # sometimes segment is wrong, will cause dead loop
                break
        else:
            break


    return planes_para, planes_label


def merge_planes(planes_para, planes_label, segment):
    """
    merge separate planes with same normal and segment label,
    especially for walls (ScanNet id 1) , floor (ScanNet id 3) and ceilings (ScanNet id 41)
    :param planes_para:
    :param planes_label:
    :param segment:
    :return:
    """

    floor = (segment == 3)
    wall = (segment == 1)
    ceiling = (segment == 41)

    planes_para, planes_label = merge_planes_in_one_seg_label(planes_para, planes_label, floor)
    planes_para, planes_label = merge_planes_in_one_seg_label(planes_para, planes_label, wall)
    planes_para, planes_label = merge_planes_in_one_seg_label(planes_para, planes_label, ceiling)

    return planes_para, planes_label


def test2():
    segment_dir = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0011_00/label-filt/"
    seg_num = len(os.listdir(segment_dir))

    for i in range(seg_num):
        print("%d.jpg" %(i))
        plane_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0011_00/plane_peac_scale4/" + \
                         str(i) + ".jpg-plane-data.txt"
        planes_para = read_plane_paramters_file(plane_filepath)

        plane_label_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0011_00/plane_peac_scale4/" + \
                               str(i) + ".jpg-plane-label.txt"
        label = read_plane_label_file(plane_label_filepath)

        segment_filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0011_00/label-filt/" + \
                           str(i) + ".png"
        segment = cv2.imread(segment_filepath, -1)
        segment = cv2.resize(segment, (640, 480))

        planes_para, planes_label = merge_planes(planes_para, label, segment)
        # print(len(planes_para))

        plane_refined_dir = "/home/xiaoxiao/disk6/ScanNet/scene0011_00/plane_refined/"

        if not os.path.exists(plane_refined_dir):
            os.mkdir(plane_refined_dir)

        planes_para, planes_label = merge_planes(planes_para, planes_label, segment)
        # print(len(planes_para))
        np.save(os.path.join(plane_refined_dir, str(i) + ".jpg-plane-data.npy"), planes_para)
        cv2.imwrite(os.path.join(plane_refined_dir, str(i) + ".jpg-plane-label.png"), np.uint8(planes_label))

        color_label = planes_label2color(planes_label, len(planes_para)+1)
        cv2.imwrite(os.path.join(plane_refined_dir, str(i) + ".jpg-plane-color-label.png"), np.uint8(color_label))

def scp_plane_files():
    destination = "/home/xiaoxiao/disk6/ScanNet/"
    for dir in os.listdir(destination):
        print(dir)
        command = "sshpass -p 'xiaoxiao' scp -r xiaoxiao@192.168.38.175:/home/xiaoxiao/disk2/xiaoxiao/dataset/ScanNet/rawData/" \
                  + dir + "/plane_peac_scale4 /home/xiaoxiao/disk6/ScanNet/" + dir + "/"
        os.system(command)


def make_soft_link():
    destination = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/"
    source = "/home/xiaoxiao/disk6/ScanNet/"

    i = 0
    for dir in os.listdir(destination):
        if i == 0:
            i += 1
            continue
        else:
            i +=1
        print(destination + dir + "/" + dir + ".txt")
        # if os.path.exists(destination + dir + "/" + dir + ".txt"):
        #     command_1 = "rm " + destination + dir + "/" + dir + ".txt"
        #     print(command_1)
        #     os.system(command_1)
        os.rename(source + dir + "/plane_errors_003", source + dir + "/planercnn_error_003")
        command1 = "ln -s " + source + dir + "/planercnn_error_003" + " " + destination + dir + "/"
        #command2 = "ln -s " + source + dir + "/planercnn_seg_003" + " " + destination + dir + "/"
        #command3 = "ln -s " + source + dir + "/planercnn_seg_color_003" + " " + destination + dir + "/"
        os.system(command1)
        #os.system(command2)
        #os.system(command3)


if __name__ == '__main__':
    # filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0000_00/plane_peac_scale4/0.jpg-plane-data.txt"
    # planes = read_plane_paramters_file(filepath)

    # filepath = "/home/xiaoxiao/disk2/ScanNet/rawData/scans/scene0000_00/plane_peac_scale4/0.jpg-plane-label.txt"
    # label = read_plane_label_file(filepath)
    make_soft_link()
    # test2()
