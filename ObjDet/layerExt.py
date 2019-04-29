# Imports
import torch
import torchvision

# Image Size
image_size = (256, 256)

# Create a dummy image
dummy_img = torch.zeros((1, 3, image_size[0], image_size[1])).float()

#print(dummy_img)

model = torchvision.models.vgg16(pretrained=True)
fe = list(model.features)

print(fe)
