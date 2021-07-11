# Multi Point-Voxel Convolution (MPVConv) for Deep Learning on Point Clouds

## Environments
- Ubuntu 20.04

- Python 3.8.3

- Pytorch 1.7.0

## Prerequisites

The code is built with following libraries (see [requirements.txt](requirements.txt)):
- Python >= 3.8
- [PyTorch](https://github.com/pytorch/pytorch) >= 1.7
- [numba](https://github.com/numba/numba) >= 0.51.2
- [numpy](https://github.com/numpy/numpy) >= 1.19.2
- [scipy](https://github.com/scipy/scipy) >= 1.5.2
- [six](https://github.com/benjaminp/six) >= 1.15.0
- [tensorboardX](https://github.com/lanpa/tensorboardX) >= 2.1
- [tqdm](https://github.com/tqdm/tqdm) >= 4.51.0
- [plyfile](https://github.com/dranjan/python-plyfile) >= 0.7.2
- [h5py](https://github.com/h5py/h5py) >= 2.10.0

## Part Segmentation (ShapeNet Part)
### Data Preparation
Download the alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) by [PointNet2](https://github.com/charlesq34/pointnet2) and unzip in `data/shapenet/shapenetcore_partanno_segmentation_benchmark_v0_normal/`.

Or run the following command to download and unzip the dataset:
```bash
./data/shapenet/download.sh
```
### Training
To train the MPVCNN model for ShapeNet Part, one can run:
 
```
python train.py [config-file] --devices [gpu-ids]
```

For example, to train the full model of MPVCNN, you can run:
```
python train.py configs/shapenet/mpvcnn/c1.py --devices 0,1
```
To train the half feature channels of MPVCNN, just run:
```
python train.py configs/shapenet/mpvcnn/c0p5.py --devices 0,1
```

**NOTE**: During training, the meters will provide rough estimations of accuracies and IoUs.

### Evaluating and Visualization
To evaluate trained models and visualize the prediction results in ShapeNet Part dataset, you can run:

```
python train.py [config-file] --devices [gpu-ids] --evaluate
```
For example, to evaluate and visualize the full model of MPVCNN, you should run:
```
python train.py configs/shapenet/mpvcnn/c1.py --devices 0 --evaluate --configs.evaluate.best_checkpoint_path runs/shapenet.mpvcnn.c1/best.pth.tar
```
The visualization results are located in `visual/shapenet/` with `_pred.obj` file. The IoUs of all predictions are located in `visual/shapenet/shapenet_iou.txt`.
### Performance

|                                                  Models                                                         |     mIoU     | 
| :-------------------------------------------------------------------------------------------------------------: | :----------: |
|  [PointNet](https://hanlab.mit.edu/projects/pvcnn/files/models/shapenet.pointnet.pth.tar)                       |     83.5     |
|  PointNet                                                                                                       |     83.7     |
|  3D-UNet                                                                                                        |     84.6     |
|  [PointNet++ SSG (Reproduce)](https://hanlab.mit.edu/projects/pvcnn/files/models/shapenet.pointnet2ssg.pth.tar) |     85.1     |
|  PointNet++ MSG                                                                                                 |     85.1     |
|  SpiderCNN                                                                                                      |     85.3     |
|  [PointNet++ MSG (Reproduce)](https://hanlab.mit.edu/projects/pvcnn/files/models/shapenet.pointnet2msg.pth.tar) |     85.3     |
|  [PVCNN](https://hanlab.mit.edu/projects/pvcnn/files/models/shapenet.pvcnn.c1.pth.tar)                          |     85.8     |
|  PointCNN                                                                                                       |     86.1     |
|  MPVCNN |**86.5** | 


## Indoor Scene Segmentation (S3DIS)
### Data Preparation
Download 3D indoor parsing dataset (**S3DIS**) [here](http://buildingparser.stanford.edu/dataset.html)  and save in `data/s3dis/Stanford3dDataset_v1.2_Aligned_Version/`.

After downloading **S3DIS** dataset, then preprocessing the downloaded data (**S3DIS**) by the method in [PointCNN](https://github.com/yangyanli/PointCNN).
The code for preprocessing the S3DIS dataset is located in `data/s3dis/prepare_data.py`, then run:
```bash
python data/s3dis/prepare_data.py -d [path to unzipped dataset dir]
```
The default output is located in `data/s3dis/pointcnn`.
### Training
To train the MPVCNN model for S3DIS, one can run:
 
```
python train.py [config-file] --devices [gpu-ids]
```

For example, to train the full model of MPVCNN, you can run:
```
python train.py configs/s3dis/mpvcnn/area5/c1.py --devices 0,1
```
To train the quarter feature channels of MPVCNN, just run:
```
python train.py configs/s3dis/mpvcnn/area5/c0p25.py --devices 0,1
```

### Evaluating and Visualization
To evaluate trained models and visualize the prediction results in S3DIS dataset, you can run:

```
python train.py [config-file] --devices [gpu-ids] --evaluate
```
For example, to evaluate and visualize the full model of MPVCNN, you should run:
```
python train.py configs/s3dis/mpvcnn/area5/c1.py --devices 0 --evaluate --configs.evaluate.best_checkpoint_path runs/s3dis.mpvcnn.area5.c1/best.pth.tar
```
The visualization results are located in `visual/s3dis/` with `_pred.obj` file.

### Performance

|                                                  Models                                                     | Overall Acc |     mIoU     | 
| :---------------------------------------------------------------------------------------------------------: | :---------: | :----------: |
|  PointNet                                                                                                   |    82.54    |     42.97    |
|  [PointNet](https://hanlab.mit.edu/projects/pvcnn/files/models/s3dis.pointnet.area5.pth.tar)    |    80.46    |     44.03    |
|  3D-UNet                                                                                                    |    85.12    |     54.93    |
|  [PVCNN](https://hanlab.mit.edu/projects/pvcnn/files/models/s3dis.pvcnn.area5.c1.pth.tar)                   |    86.47    |     56.64    |
|  PointCNN                                                                                                   |    85.91    |     57.26    |
|  [PVCNN++](https://hanlab.mit.edu/projects/pvcnn/files/models/s3dis.pvcnn2.area5.c1.pth.tar)                |    87.48    |     59.02    |
|  MPVCNN|87.75|58.63|
|MPVCNN++|**89.31**|**61.51**|

## 3D Object Detection (KITTI)
### Data Preparation
For Frustum-PointNet backbone, we follow the data pre-processing in [Frustum-Pointnets](https://github.com/charlesq34/frustum-pointnets).
One should first download the ground truth labels `data_object_label_2.zip` [here](http://www.cvlibs.net/download.php?file=data_object_label_2.zip) and unzip it in `data/kitti/ground_truth`, then download `frustum_data.zip` [here](https://shapenet.cs.stanford.edu/media/frustum_data.zip) and save in `data/kitti/frustum/frustum_data/`.

Or just run:
```bash
unzip data_object_label_2.zip
mv training/label_2 data/kitti/ground_truth
./data/kitti/frustum/download.sh
```

### Training
To train the MPVCNN model for KITTI, one can run:
 
```
python train.py [config-file] --devices [gpu-ids]
```

For example, to train F-MPVCNN, you can run:
```
python train.py configs/kitti/frustum/mpvcnne.py --devices 0,1
```

### Evaluation

To evaluate the trained models in KITTI dataset, you can run:
```
python train.py configs/kitti/frustum/mpvcnne.py --devices 0 --evaluate --configs.evaluate.best_checkpoint_path runs/kitti.frustum.mpvcnne/best.pth.tar
```

### Performance

|                                                   Models                                                             |        Car        |        Car        |        Car        |     Pedestrian    |     Pedestrian    |     Pedestrian    |      Cyclist      |      Cyclist      |      Cyclist      |
|:--------------------------------------------------------------------------------------------------------------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
|                                                                                                                      |        Easy       |      Moderate     |        Hard       |        Easy       |      Moderate     |        Hard       |        Easy       |      Moderate     |        Hard       |
| Frustum PointNet                                                                                                     |       83.26       |       69.28       |       62.56       |         -         |         -         |         -         |         -         |         -         |         -         |
| [Frustum PointNet (Reproduce)](https://hanlab.mit.edu/projects/pvcnn/files/models/kitti.frustum.pointnet.pth.tar)    |   85.24 (85.17)   |   71.63 (71.56)   |   63.79 (63.78)   |   66.44 (66.83)   |   56.90 (57.20)   |   50.43 (50.54)   |   77.14 (78.16)   |   56.46 (57.41)   |   52.79 (53.66)   |
| Frustum PointNet++                                                                                                   |       83.76       |       70.92       |       63.65       |       70.00       |       61.32       |       53.59       |       77.15       |       56.49       |       53.37       |
| [Frustum PointNet++ (Reproduce)](https://hanlab.mit.edu/projects/pvcnn/files/models/kitti.frustum.pointnet2.pth.tar) |   84.72 (84.46)   |   71.99 (71.95)   |   64.20 (64.13)   |   68.40 (69.27)   |   60.03 (60.80)   |   52.61 (53.19)   |   75.56 (79.41)   |   56.74 (58.65)   | 53.33 (**54.82**) |
| [Frustum PVCNN](https://hanlab.mit.edu/projects/pvcnn/files/models/kitti.frustum.pvcnne.pth.tar)         | 85.25 | 72.12 | 64.24  | 70.60 | 61.24  | 56.25  | 78.10| 57.45 | 53.65  |
| Frustum MPVCNN| **85.66**|**72.63** |**64.62**|**71.12**|**62.34**|**57.13**|**79.85**|**58.28**|**54.62**|

## Reference
[PVCNN](https://github.com/mit-han-lab/pvcnn/).

[PointCNN](https://github.com/yangyanli/PointCNN/).

[PointNet2](https://github.com/charlesq34/pointnet2).

[Pointnet2_PyTorch](https://github.com/erikwijmans/Pointnet2_PyTorch).

[Frustum-Pointnets](https://github.com/charlesq34/frustum-pointnets).

[kitti-object-eval-python](https://github.com/traveller59/kitti-object-eval-python).
