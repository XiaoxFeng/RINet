# Weakly Supervised Rotation-Invariant Aerial Object Detection Network
By Xiaoxu Feng, Xiwen Yao, Gong Cheng, Junwei Han
## Citation

## Overview
![Overview](https://github.com/XiaoxFeng/RINet/blob/main/Overview.jpg)

The code will be released soon.
## Requirements
* python == 3.6 <br>
* Cuda == 9.0 <br>
* Pytorch == 0.4.1 <br>
* torchvision == 0.2.1 <br>
* Pillow <br>
* sklearn <br>
* opencv <br>
* scipy <br>
* cython <br>
* GPU: GeForce RTX 2080Ti | Tesla V100
## Installation
1. Clone the RINet repository
```bash
git clone https://github.com/XiaoxFeng/RINet.git
``` 
2. Compile
```bash
cd RINet/lib
bash make.sh
```
3.Download the VOCdevkit and renamed it as VOCdevkit2007
```bash
cd RINet/data/
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
```
4. Download the training, validation, test data from [NWPU](https://onedrive.live.com/?authkey=%21ADaUNysmiFRH4eE&cid=5C5E061130630A68&id=5C5E061130630A68%21115&parId=5C5E061130630A68%21113&action=locate) and [DIOR](https://drive.google.com/drive/folders/1UdlgHk49iu6WpcJ5467iT-UqNPpx__CC)
5. Extract all of datasets into one directory named VOCdevkit2007
6. Download pretrained ImageNet weights from [here](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ), and put it in the data/imagenet_weights/
7. Download selective search proposals from [NWPU](https://drive.google.com/file/d/1VnmUDPomgTgmHvH3CemFOIWTLuVR5f-t/view?usp=sharing) and [DIOR](https://drive.google.com/file/d/1wbivkAxqBQB4vAX0APmVzIOhuawHpsPV/view?usp=sharing), and put it in the data/selective_search_data/
## Training and Testing
Train a vgg16 Network on DIOR trainval
```bash
bash experiments/scripts/train_faster_rcnn.sh 0 DIOR vgg16
```
Test a vgg16 Network on DIOR test
```bash
bash experiments/scripts/test_faster_rcnn.sh 0 DIOR vgg16
'''
## Acknowledgement
We borrowed code from [MLEM](https://github.com/vasgaowei/pytorch_MELM), [PCL](https://github.com/ppengtang/pcl.pytorch), and [Faster-RCNN](https://github.com/jwyang/faster-rcnn.pytorch).
