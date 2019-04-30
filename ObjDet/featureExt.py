# Imports
import torch
import torchvision
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Debug flag, set to 1 to get debug messages
__DEBUG__ = 0

image_size = (256, 256)
# Sample Black background image
image = torch.zeros((1, 3, image_size[0], image_size[1])).float()

if __DEBUG__:
    print("size of image tensor: %s " %(image.size()))

# Generate Sample Bounding Box
bbox = torch.FloatTensor([[20, 30, 200, 150],[150, 200, 220, 250]]) #[Ymin, Xmin, Ymax, Xmax] format
labels = torch.LongTensor([6, 8])
sub_sample = 16

# Squeeze to remove the first dimention of tensor (convert from 4d to 3d I think)
pil_image = transforms.ToPILImage()(image.squeeze())

#plt.imshow(pil_image)
if __DEBUG__:
    pil_image.show()

# Create a dummy image
dummy_img = torch.zeros((1, 3, image_size[0], image_size[1])).float()

#print(dummy_img)

# Load vgg16
model = torchvision.models.vgg16(pretrained=True)

# List out all the features
fetrs = list(model.features)

# Pass dummy image through layers to check for layer whose output matches the required feature map size
req_fetrs = []
# Clone dummy image and pass it through all layers and check for layer output size
clone_img = dummy_img.clone()
for lyr in fetrs:
    clone_img = lyr(clone_img)
    if clone_img.size()[2] < 256//16:
        break
    req_fetrs.append(lyr)
    out_channels = clone_img.size()[1]

if __DEBUG__:
    print("Length of required features: ", len(req_fetrs))
    print("Number of Out channels: ", out_channels)
    
# Convert required features into sequential module
frcnn_fe = nn.Sequential(*req_fetrs)

# Using frcnn as backend compute features for dummy image
out_map = frcnn_fe(image)

if __DEBUG__:
    print("Out map size is: ", out_map)

# Creating Anchors
# ----------------------
# Define 3 ratio and scales that we will be using
ratio = [0.5, 1, 2]
anchor_scales = [8, 16, 32]

# Number of Ratios and anchor scales
n_ratio = len(ratio)
n_scales = len(anchor_scales)

# Base for the anchor
anchor_base = np.zeros((n_ratio * n_scales, 4), dtype=np.float32)

if __DEBUG__:
    print("anchor base is: \n", anchor_base)

# Define center for base anchor
center_y = sub_sample / 2.
center_x = sub_sample / 2.

if __DEBUG__:
    print("Center for base anchor is: (%s, %s) " %(center_x, center_y))

# Generating Anchos for first feature map pixel
# Iterate through all ratios and scales
for i in range(n_ratio):
    for j in range(n_scales):
        h = sub_sample * anchor_scales[j] * np.sqrt(ratio[i])
        w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratio[i])

        index = i * n_scales + j

        anchor_base[index, 0] = center_y - h / 2. #y_min
        anchor_base[index, 1] = center_x - w / 2. #x_min
        anchor_base[index, 2] = center_y + h / 2. #y_max
        anchor_base[index, 3] = center_x + w / 2. #x_max

if __DEBUG__:
    print("anchor bases: \n", anchor_base)
    print("Negative anchors represent the ones that are out of the image boundaries")

# Generationg anchors for all feature map pixels
feature_size = (image_size[0] // sub_sample)
# 16 sub_samples in feature map where each has dimension 16*16
center_x = np.arange(sub_sample, (feature_size +  1) * 16, 16)
center_y = np.arange(sub_sample, (feature_size +  1) * 16, 16)

# Generation Centers
center = np.zeros((len(center_x) * len(center_y), 2))
index = 0
for x in range(len(center_x)):
    for y in range(len(center_y)):
        center[index, 1] = center_x[x] - int(sub_sample / 2)
        center[index, 0] = center_y[y] - int(sub_sample / 2)
        index += 1

# Generating anchors for above generated centers
num_anchors_per_pixel = n_ratio * n_scales
anchors = np.zeros(((feature_size * feature_size * num_anchors_per_pixel), 4))

index = 0
for c in center:
    center_y, center_x = c
    for i in range(n_ratio):
        for j in range(n_scales):
            h = sub_sample * anchor_scales[j] * np.sqrt(ratio[i])
            w = sub_sample * anchor_scales[j] * np.sqrt(1. / ratio[i])

            anchors[index, 0] = center_y - h / 2.
            anchors[index, 1] = center_x - w / 2.
            anchors[index, 2] = center_y + h / 2.
            anchors[index, 3] = center_x + w / 2.
            index += 1

if __DEBUG__:
    print("Total anchors size is: ", anchors.shape)


