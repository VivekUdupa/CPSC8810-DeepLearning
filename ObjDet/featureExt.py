# Imports
import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image

# Debug flag, set to 1 to get debug messages
__DEBUG__ = 0

imageSize = (256, 256)
# Sample Black background image
image = torch.zeros((1, 3, imageSize[0], imageSize[1])).float()

if __DEBUG__:
    print("size of image tensor: %s " %(image.size()))

# Generate Sample Bounding Box
bbox = torch.FloatTensor([[20, 30, 200, 150],[150, 200, 220, 250]]) #[Ymin, Xmin, Ymax, Xmax] format
labels = torch.LongTensor([6, 8])
sub_sample = 16

pil_image = transforms.ToPILImage()(image.squeeze())

#plt.imshow(pil_image)
pil_image.show()
