# Weakly Supervised Rotation-Invariant Aerial Object Detection Network
By Xiaoxu Feng, Xiwen Yao, Gong Cheng, Junwei Han

The code will be released soon.
## Citation
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
2. Compile
3. Download the training, validation, test data and VOCdevkit
4. Extract all of these tars into one directory named VOCdevkit
5. Download pretrained ImageNet weights from [here](https://www.dropbox.com/s/s3brpk0bdq60nyb/vgg16_caffe.pth?dl=0), and put it in the data/imagenet_weights/
6. Download selective search proposals from [NWPU](https://drive.google.com/file/d/1VnmUDPomgTgmHvH3CemFOIWTLuVR5f-t/view?usp=sharing) and [DIOR](https://drive.google.com/file/d/1wbivkAxqBQB4vAX0APmVzIOhuawHpsPV/view?usp=sharing), and put it in the data/selective_search_data/
