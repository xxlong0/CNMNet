import os

source = "/home/xiaoxiao/disk7/ScanNet_mini/mini_rawData"

for scene in sorted(os.listdir(source)):
    lg_normal_dir = os.path.join(source, scene, "lg_normal")
    # command = "sshpass -p lxx5419 scp -r " + source + "/" + dir + "/" + seq + " xxlong@gpugate2.cs.hku.hk:/userhome/35/xxlong/dataset/7_scenes/" + dir + "/"
    command = "sshpass -p wang123 scp -r " + source + "/" + scene + "/" + "lg_normal" + \
              " jpwang@gpugate2.cs.hku.hk:/userhome/35/jpwang/dataset/ScanNet/home/xiaoxiao/mnt2/ScanNet/mini_rawData/" + scene + "/"
    print(command)
    os.system(command)
