# Importing libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys # For command Line arguments
import os
from shutil import copyfile
from detection_Model import CNNModel
from PIL import Image

# loading the input test image
if(len(sys.argv)<2): #If test image path is not mentioned
    sys.exit("Please specify an image to test")
else:
    test_img_filename = sys.argv[1]

img_size = (256, 256)
n_cnn = 3
conv_size = int( img_size[0]/(2**n_cnn) )
test_img  = "./TestData/test/"
test_img1  = "./TestData"
Model = "./Model"

# Hyperparameter initialization
batch_size      = 1
  
# Define the transformation
transform = transforms.Compose( [transforms.Resize(img_size),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                 ])


# Testing dataset
test_dataset = Image.open(test_img_filename)
test_loader = transform(test_dataset)

# Image parameters
n_class         = 10

model = CNNModel(conv_size, n_class).cuda()
model.load_state_dict(torch.load('./Model/model.pth'))
model.eval().cuda()

def ten_to_str(x):
	""" Function to convert tensor label to a string """
	str_label = ["gossiping", "isolation", "laughing", "nonbullying", "pullinghair", "punching", "quarrel", "slapping", "stabbing", "strangle"]
	return str_label[x]

# Testing the model
with torch.no_grad():
    images = Variable(test_loader, requires_grad=True)
    images = images.unsqueeze(0)
    images = images.cuda()
    outputs = model(images)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.item()
    print("{}".format(ten_to_str(predicted)))
