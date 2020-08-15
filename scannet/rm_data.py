import os

source = "~/dataset/ScanNet/home/xiaoxiao/mnt2/ScanNet/mini_rawData/"

for scene in os.listdir(source):
    normal_dir = os.path.join(source, scene, "normal")
    command = "rm -rf "+ normal_dir
    print(command)
    os.system(command)