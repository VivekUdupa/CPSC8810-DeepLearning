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

# Labeling the anchors
#[Ymin, Xmin, Ymax, Xmax] format
bbox = np.asarray([[20, 30, 200, 150],[150, 200, 220, 250]], dtype=np.float32) 
labels = np.asarray([6, 8])

# Find the index of the anchors that are inside the image boundary
index_inside = np.where(
                (anchors[:, 0] >= 0) &
                (anchors[:, 1] >= 0) &
                (anchors[:, 2] <= image_size[1]) &
                (anchors[:, 3] <= image_size[0]) 
                )[0]

if __DEBUG__:
    print("anchors that are insdie the image are: \n", index_inside)
    print("\nNumber of anchors inside the image boundary: ", index_inside.shape)

# Make a label array and fill it with -1
label = np.empty((len(index_inside), ), dtype=np.int32)
label.fill(-1)

if __DEBUG__:
    print("Created Label size is %s and index inside size is %s" %(label.size, index_inside.size))

# Array with valid anchor boxes
anchor_valid = anchors[index_inside]

if __DEBUG__:
    print("Valid anchor box shape is: ", anchor_valid.shape)

# Calculate IoU for valid anchor boxes
ious = np.empty((len(anchor_valid), 2), dtype=np.float32)
if __DEBUG__:
    print("Bounding Boxes are : \n", bbox)

for num1, i in enumerate(anchor_valid):
    # ymin, xmin, ymax, xmax format for anchors
    ya1, xa1, ya2, xa2 = i
    # anchor area = height * width
    area_anchor = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        area_box = (yb2 - yb1) * (xb2 - xb1)

        intersection_x1 = max([xb1, xa1])
        intersection_y1 = max([yb1, ya1])
        intersection_x2 = min([xb2, xa2])
        intersection_y2 = min([yb2, ya2])

        # Check for intersection
        if (intersection_x1 < intersection_x2) and (intersection_y1 < intersection_y2):
            area_intersection = (intersection_y2 - intersection_y1) * (intersection_x2 - intersection_x1)
            #intersection over union
            iou = area_intersection / (area_anchor + area_box - area_intersection)
        else:
            # In case of No overlap/ intersection
            iou = 0.
        
        ious[num1, num2] = iou

if __DEBUG__:
    print("all the iou count: ", ious.shape)
        
# Case-1
# Highest IoU for each gt and corrosponding anchor
# Location of max Iou
gt_argmax_ious = ious.argmax(axis=0)
if __DEBUG__:
    print("Indices of MAX IoU: ", gt_argmax_ious)

# Value of Max IoU
gt_max_ious = ious[gt_argmax_ious, np.arange(ious.shape[1])]
if __DEBUG__:
    print("Values of MAX IoU: ", gt_max_ious)

# Case-2
# Highest Iou In between every anchor
argmax_ious = ious.argmax(axis=1)
if __DEBUG__:
    print("shape of argmax_ious: ", argmax_ious.shape)
    print("MAX Iou indices for every anchor: ", argmax_ious)

max_ious = ious[np.arange(len(index_inside)), argmax_ious]
if __DEBUG__:
    print("MAX IoU values: ", max_ious)

# Anchor Box that has the HIGHEST IoU with GT
gt_argmax_ious = np.where(ious == gt_max_ious)[0]
if __DEBUG__:
    print("Ultimate MAX IoU: ", gt_argmax_ious)

# Assigning Labels which helps to compute Loss
# Defining thresholds
pos_thres = 0.7
neg_thres = 0.3

# Assign negative label (0) to all anchors that have IoU < 0.3
label[max_ious < neg_thres] = 0

# Assign positive label (1) to anchor boxes with  highest IoU with GT
label[gt_argmax_ious] = 1

# Assign positive label (1) to anchor boxes with IoU > 0.7
label[max_ious >= pos_thres] = 1


# ==============================================================================
#                               TRAINING RPN
# ==============================================================================

# Define positive and negative anchor sample parameters
num_samples = 128
pos_ratio = 0.5
num_pos = pos_ratio * num_samples

# Picking Positive Samples
pos_index = np.where(label == 1)[0]

if len(pos_index) > num_pos:
    disable_index = np.random.choice(pos_index, size=(len(pos_index) - num_pos), replace=False)
    label[disable_index] = -1

# Picking Negative Samples
num_neg = num_samples * np.sum(label == 1) 
neg_index = np.where(label == 0)[0]

if len(neg_index) > num_neg:
    disable_index = np.random.choice(neg_index, size=(len(neg_index) - num_neg), replace=False)
    label[disable_index] = -1

# GT with MAX IoU for each anchor
max_iou_bbox = bbox[argmax_ious]

# Convert [ymin, xmin, ymax, xmax] format to [center_y, center_x, h, w] format

height = anchor_valid[:, 2] - anchor_valid[:, 0]
width  = anchor_valid[:, 3] - anchor_valid[:, 1]
center_y  = anchor_valid[:, 0] + 0.5 * height
center_x  = anchor_valid[:, 1] + 0.5 * width

base_height = max_iou_bbox[:, 2] - max_iou_bbox[:, 0]
base_width = max_iou_bbox[:, 3] - max_iou_bbox[:, 1]
base_center_y = max_iou_bbox[:, 0] + 0.5 * base_height
base_center_x = max_iou_bbox[:, 1] + 0.5 * base_width

# Find the locations
eps = np.finfo(height.dtype).eps
height = np.maximum(height, eps)
width = np.maximum(width, eps)

dy = (base_center_y - center_y) / height
dx = (base_center_x - center_x) / width
dh = np.log(base_height / height)
dw = np.log(base_width / width)

anchor_locs = np.vstack((dy, dx, dh, dw)).transpose()
if __DEBUG__:
    print("Shape of anchor locations", anchor_locs.shape)
    print("Anchor locations", anchor_locs)

# Final Labels
anchor_labels = np.empty((len(anchors),), dtype=label.dtype)
anchor_labels.fill(-1)
anchor_labels[index_inside] = label

# Final Locations
anchor_locations = np.empty((len(anchors),) + anchors.shape[1:], dtype=anchor_locs.dtype)
anchor_locations.fill(0)
anchor_locations[index_inside, :] = anchor_locs

# =============================================================================================
#       Region Proposal Network
# =============================================================================================

mid_channels = 512
in_channels = 512
n_anchor = 9

conv1 = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)

# Bounding Box Regressor network
reg_layer = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)

# Classifier network
cls_layer = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)

# Initialization
# convolution sliding layer
conv1.weight.data.normal_(0, 0.01)
conv1.bias.data.zero_()

# Regression layer
reg_layer.weight.data.normal_(0, 0.01)
reg_layer.bias.data.zero_()

# classification layer
cls_layer.weight.data.normal_(0, 0.01)
cls_layer.bias.data.zero_()

# Training
# ---------------

x = conv1(out_map)
pred_anchor_locs = reg_layer(x)
pred_cls_scores = cls_layer(x)

if __DEBUG__:
    print("predicted class score shape: ", pred_cls_scores.shape)
    print("predicted anchor location shape: ", pred_anchor_locs.shape)

# Rearrange the tensors to align with anchor targets
pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
if __DEBUG__:
    print("Rearranged anchor location shape: ", pred_anchor_locs.shape)

pred_cls_scores = pred_cls_scores.permute(0, 2, 3, 1).contiguous()
if __DEBUG__:
    print("Rearranged class score shape: ", pred_cls_scores.shape)

# Calculate the Objectness score
objectness_score = pred_cls_scores.view(1, 16, 16, 9, 2)[:, :, :, :, 1].contiguous().view(1, -1)
if __DEBUG__:
    print("Objectness Score shape: ", objectness_score.shape)

pred_cls_scores  = pred_cls_scores.view(1, -1, 2)
if __DEBUG__:
    print("predicted class score shape final: ", pred_cls_scores.shape)

# =========================================================
# Generationg Proposals
# =========================================================

# Define parameters for training and testing

nms_thresh = 0.7 # Non- Maximum Supression Threshold
n_train_pre_nms = 12000 # number of bboxes before nms during training
n_train_post_nms = 2000 # number of bboxes after nms during training
n_test_pre_nms = 6000 # number of bboxes before nms during testing
n_test_post_nms = 300 # number of bboxes after nms during testing
min_size = 16 # minimum height of the object required to create a proposal

# Convert anchors from [ymin, xmin, ymax, xmax] to [center_Y, center_x, h, w] format
anc_height = anchors[:, 2] - anchors[:, 0]
anc_width = anchors[:, 3] - anchors[:, 1]
anc_ctr_y = anchors[:, 0] + 0.5 * anc_height
anc_ctr_x = anchors[:, 1] + 0.5 * anc_width

# Convert prediction locations and objectness score to numpy array
pred_anchor_locs_numpy = pred_anchor_locs[0].data.numpy()
objectness_score_numpy = objectness_score[0].data.numpy()

dy = pred_anchor_locs_numpy[:, 0::4]
dx = pred_anchor_locs_numpy[:, 1::4]
dh = pred_anchor_locs_numpy[:, 2::4]
dw = pred_anchor_locs_numpy[:, 3::4]

ctr_y = dy * anc_height[:, np.newaxis] + anc_ctr_y[:, np.newaxis]
ctr_x = dx * anc_width[:, np.newaxis] + anc_ctr_x[:, np.newaxis]
h = np.exp(dh) * anc_height[:, np.newaxis]
w = np.exp(dw) * anc_width[:, np.newaxis] 

# Region of Interest 
roi = np.zeros(pred_anchor_locs_numpy.shape, dtype=pred_anchor_locs_numpy.dtype)
roi[:, 0::4] = ctr_y - 0.5 * h
roi[:, 1::4] = ctr_x - 0.5 * w
roi[:, 2::4] = ctr_y + 0.5 * h
roi[:, 3::4] = ctr_x + 0.5 * w

# Clip the predicted boxes to the image size
roi[:, slice(0, 4, 2)] = np.clip(roi[:, slice(0, 4, 2)], 0, image_size[0])
roi[:, slice(1, 4, 2)] = np.clip(roi[:, slice(1, 4, 2)], 0, image_size[1])

if __DEBUG__:
    print("Region of interest: ", roi)

# Remove predicted boxes with either height of width < threshold
hs = roi[:, 2] - roi[:, 0]
ws = roi[:, 3] - roi[:, 1]
keep = np.where((hs >= min_size) & (ws >= min_size))[0]
roi = roi[keep, :]
score = objectness_score_numpy[keep]

# Sort (proposal, score) in descending order
order = score.ravel().argsort()[::-1]

# Take top values?
order = order[:n_train_pre_nms]
roi = roi[order, :]

# Calculate Region Proposals
y1 = roi[:, 0]
x1 = roi[:, 1]
y2 = roi[:, 2]
x2 = roi[:, 3]

area = (x2 - x1 + 1) * (y2 - y1 + 1)
order = score.argsort()[::-1]

keep = []

while order.size > 0:
    i = order[0]
    xx1 = np.maximum(x1[i], x1[order[1:]])
    yy1 = np.maximum(y1[i], y1[order[1:]])
    xx2 = np.minimum(x2[i], x2[order[1:]])
    yy2 = np.minimum(y2[i], y2[order[1:]])

    w = np.maximum(0.0, xx2 - xx1 + 1)
    h = np.maximum(0.0, yy2 - yy1 + 1)
    inter = w * h

    ovr = inter / (area[i] + area[order[1:]] - inter)

    inds = np.where(ovr <= nms_thresh)[0]
    order = order[inds + 1]

keep = keep[:n_train_post_nms]
roi = roi[keep]

# Proposal Targets
n_sample = 128
pos_ratio = 0.25
pos_iou_thresh = 0.5
neg_iou_thresh_hi = 0.5
neg_iou_thresh_lo = 0.0

# Calculate IoU
ious = np.empty((len(roi), 2), dtype=np.float32)
ious.fill(0)
for num1, i in enumerate(roi):
    ya1, xa1, ya2, xa2 = i
    anchor_area = (ya2 - ya1) * (xa2 - xa1)
    for num2, j in enumerate(bbox):
        yb1, xb1, yb2, xb2 = j
        box_area = (yb2 - yb1) * (xb2 - xb1)

        inter_x1 = max([xb1, xa1])
        inter_y1 = max([yb1, ya1])
        inter_x2 = min([xb2, xa2])
        inter_y2 = min([yb2, ya2])

        if(inter_x1 < inter_x2) and (inter_y1 < inter_y2): 
            inter_area = (inter_y2 - inter_y1) * (inter_x2 - inter_x1)
            iou = inter_area / (anchor_area + box_area - inter_area)
        else:
            iou = 0.

        ious[num1, num2] = iou
print(ious.shape)
print(roi)
