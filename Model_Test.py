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
from Detection_Model import CNNModel

# loading the input test image
test_img_filename = sys.argv[1]

img_size = (256,256)
conv_size = int( img_size[0]/4 )
test_img  = "./TestData/test/"
test_img1  = "./TestData"
Model = "./Model"

# Hyperparameter initialization
batch_size      = 1

if not os.path.exists(test_img):
    os.makedirs(test_img)
    
# clear the contents of the directory
for the_file in os.listdir(test_img):
    file_path = os.path.join(test_img, the_file)
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        #elif os.path.isdir(file_path): shutil.rmtree(file_path)
    except Exception as e:
        print(e)


copyfile(test_img_filename, test_img+test_img_filename)
    
# Define the transformation
transform = transforms.Compose( [transforms.Resize(img_size),
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                 ])


# Testing dataset
test_dataset = datasets.ImageFolder(root=test_img1, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# Image parameters
img_size = (256,256)
conv_size = int( img_size[0]/4 )
n_class         = 9

model = CNNModel(conv_size, n_class).cuda()
model.load_state_dict(torch.load('./Model/mark1.pth'))
model.eval().cuda()

def ten_to_str(x):
	""" Function to convert tensor label to a string """
	value = x.data[0] #Convert to data
	str_label = ["gossiping", "isolation", "laughing", "pullinghair", "punching", "quarrel", "slapping", "stabbing", "strangle"]
	return str_label[value]

# Testing the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        #if (predicted!=labels):
       	print("predicted: {} | Actual: {}, total: {} ".format(ten_to_str(predicted), ten_to_str(labels), total))

print('Test Accuracy of the model on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))

