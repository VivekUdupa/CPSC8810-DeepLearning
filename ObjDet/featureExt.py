# Imports
import torch
import torchvision
import torch.nn as nn
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
    print("Out map size is: ", out_map.size())
