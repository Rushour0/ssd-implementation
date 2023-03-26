# SSD Implmentation using PyTorch

---

## Introduction

---

This repository contains the implementation of SSD using PyTorch. The implementation is based on the paper [Very Deep Convolutional Networks for Large-Scale Image Recognition ](https://arxiv.org/pdf/1409.1556.pdf) by Karen Simonyan and Andrew Zisserman. Other references used to understand the original architecture of VGG-16D are: [GFG VGG-16 Model](https://www.geeksforgeeks.org/vgg-16-cnn-model/), [VGG and ImageNet](https://conx.readthedocs.io/en/latest/VGG16%20and%20ImageNet.html)

### What is SSD?

---

SSD is a Single Shot MultiBox Detector, which is a state-of-the-art object detection framework. It is a deep learning model that can be used to detect objects in images and videos. It is based on the Single Shot Detector (SSD) framework, which uses a single deep neural network to predict bounding boxes and class scores for multiple objects in a single image. SSD is a popular object detection framework that is used in many computer vision applications, such as self-driving cars, face detection, and object tracking.

## Implementation

---

The backbone model network VGG-16 is used for feature extraction. The feature maps are then passed through a series of convolutional layers to generate the final output. The final output is a set of bounding boxes and class scores for each object in the image.

The following has been used as a reference for the implementation of the VGG-16D model:

[![Table](/readme/ConvNet%20-%20Table.jpeg)](https://arxiv.org/pdf/1409.1556.pdf)

Here, instead of using Conv3D, Conv2D is used to extract the features from the input image. The input image is passed through a series of convolutional layers to generate the feature maps. The feature maps are then passed through a series of convolutional layers to generate the final output. The final output is a set of bounding boxes and class scores for each object in the image.

The difference won't be much for image training, since there is no temporal context needed between independent images. However, for video training, the Conv3D will be more efficient, since it can extract the features from the input video with the temporal context.

### Conv2D v/s Conv3D

Reference : https://stats.stackexchange.com/questions/296679/what-does-kernel-size-mean

<table>
<tr>
<td>Conv2D</td>
<td>Conv3D</td>
</tr>
<tr>
<td><img src="/readme/conv2d-explained.gif" alt="Conv2D" width="400"/></td>
<td><img src="/readme/conv3d-explained.gif" alt="Conv3D" width="400"/></td>
</tr>
</table>

## Networks

---
