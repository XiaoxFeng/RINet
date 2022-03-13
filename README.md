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

'''

  jhjhj

'''

5. Compile
6. Download the training, validation, test data and VOCdevkit
7. Extract all of these tars into one directory named VOCdevkit
8. Download pretrained ImageNet weights from [here](https://drive.google.com/drive/folders/0B1_fAEgxdnvJSmF3YUlZcHFqWTQ), and put it in the data/imagenet_weights/
9. Download selective search proposals from [NWPU](https://drive.google.com/file/d/1VnmUDPomgTgmHvH3CemFOIWTLuVR5f-t/view?usp=sharing) and [DIOR](https://drive.google.com/file/d/1wbivkAxqBQB4vAX0APmVzIOhuawHpsPV/view?usp=sharing), and put it in the data/selective_search_data/
## Acknowledgement
We borrowed code from [MLEM](https://github.com/vasgaowei/pytorch_MELM), [PCL](https://github.com/ppengtang/pcl.pytorch), and [Faster-RCNN](https://github.com/jwyang/faster-rcnn.pytorch).
