# Occlusion-Aware Depth Estimation with Adaptive Normal Constraints
[Xiaoxiao Long](https://www.xxlong.site),[Lingjie Liu](https://lingjie0206.github.io), [Christian Theobalt](http://people.mpi-inf.mpg.de/~theobalt/),[Wenping Wang](https://i.cs.hku.hk/~wenping), ECCV 2020

<p align="center">
    <img src="./docs/images/teaser.png" alt="Image" width="512"  />
</p>

## Introduction
We present a new learning-based method for multi-frame depth estimation from a color video, which is a fundamental problem in scene understanding, robot navigation or handheld 3D reconstruction. While recent learning-based methods estimate depth at high accuracy, 3D point clouds exported from their depth maps often fail to preserve important geometric feature (e.g., corners, edges, planes) of man-made scenes. Widely-used pixel-wise depth errors do not specifically penalize inconsistency on these features. These inaccuracies are particularly severe when subsequent depth reconstructions are accumulated in an attempt to scan a full environment with man-made objects with this kind of features. Our depth estimation algorithm therefore introduces a Combined Normal Map (CNM) constraint, which is designed to better preserve high-curvature features and global planar regions.
In order to further improve the depth estimation accuracy, we introduce a new occlusion-aware strategy that aggregates initial depth predictions from multiple adjacent views into one final depth map and one occlusion probability map for the current reference view. Our method outperforms the state-of-the-art in terms of depth estimation accuracy, and preserves essential geometric features of man-made indoor scenes much better than other algorithms.

If you find this project useful for your research, please cite: 
```
@article{long2020occlusion,
  title={Occlusion-Aware Depth Estimation with Adaptive Normal Constraints},
  author={Long, Xiaoxiao and Liu, Lingjie and Theobalt, Christian and Wang, Wenping},
  journal={ECCV},
  year={2020}
}
```

## How to use

### Environment
The environment requirements are listed as follows:
- Pytorch>=1.2.0
- CUDA 10.0 
- CUDNN 7

### Preparation
* Check out the source code 

    ```git clone https://github.com/xxlong0/CNMNet.git && cd CNMNet```
* Install dependencies 

* Prepare training/testing datasets
    * [ScanNet](http://www.scan-net.org/) : Due to license of ScanNet, please follow the instruction of ScanNet and download the raw dataset.
    * [7scenes](https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/)
    * [plane segmentation of ScanNet](https://github.com/NVlabs/planercnn): Please follow the instruction and download the plane segmentation annotations.
### Training
* Before start training, need to clean the plane segementation annotations and do data preprocessing.
```bash
# train the DepthNet without refinement
python train.py train with dataset.batch_size=5 dataset.root_dir=/path/dataset dataset.list_filepath=./scannet/train_plane_view3_scans0_999_interval2_error01.txt dataset.image_width=256 dataset.image_height=192 k_size=9

# train whole model with refinement (one reference image and two source images)
python train.py train_refine with dataset.batch_size=5 dataset.root_dir=/path/dataset dataset.list_filepath=./scannet/train_plane_view3_scans0_999_interval2_error01.txt dataset.image_width=256 dataset.image_height=192 k_size=9
```

### Testing
* predict disparity and convert to depth (more accurate for near objects, reported in paper): download [pretrained model](https://drive.google.com/file/d/1tQ2mL0o8kTL5HsencJc7NSuolwtqCcgu/view?usp=sharing)

```bash
# evaluate the DepthNet without refinement
python eval.py eval with dataset.batch_size=5 dataset.root_dir=/path/dataset dataset.list_filepath=./scannet/train_plane_view3_scans0_999_interval2_error01.txt dataset.image_width=256 dataset.image_height=192 k_size=9

# evaluate whole model with refinement (one reference image and two source images)
python eval.py eval_refine with dataset.batch_size=5 dataset.root_dir=/path/dataset dataset.image_width=256 dataset.image_height=192 k_size=9
```

### Acknowledgement
The code partly relies on code from [MVDepthNet](https://github.com/HKUST-Aerial-Robotics/MVDepthNet)
