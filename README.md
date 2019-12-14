# unet_voc
PyTorch implementation of U-Net for PASCAL VOC 2012 image segmentation task.

# U-Net
U-Net is a neural network architecture for image segmentation tasks.
It has "U-shaped" architecture which consists of encoder ("contracting path"), decoder ("expanding path"), and shortcut path between them.
See the original paper for more detail information.

[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)

# Usage
## Download dataset
Download and extract PASCAL VOC 2012 train/val dataset in `unet_voc/`.
Train and val images should be placed in `./VOCdevkit/VOC2012/JPEGImages/`.

We randomly select 2713 image as a training set from 2913 images with segmentation data of PASCAL VOC 2012 train/val set.
The other 200 images are used as a test set.
The image file names are listed in `unet_voc/ImageSets/*.txt`.

## Training
`python3 train.py  tag_name  epochs`

`tag_name` is an arbitrary string that identifies the trained model.
`epochs` is an integer that specifies training epoch.

Trained models are generated every 10 epochs and are saved as `ckpt/tag_name/model_epoch*.pth`

## Test
`python3 test.py ckpt/tag_name/model_epoch*.pth`

Accuracy of each image and their average is printed in stdout.
Generated segmentation maps are saved in `./SegImage/`.
