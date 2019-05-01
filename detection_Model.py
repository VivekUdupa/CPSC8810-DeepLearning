# Importing libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import sys # For command Line arguments

#Defining the CNN
class CNNModel(nn.Module):
    """ A CNN Model for image classification """
    
    def __init__(self,image_size, op_size):
        """ CNN layer to process the image"""
        super(CNNModel, self).__init__() # Super is used to refer to the base class, i.e nn.Module

        # Convolution Layer 1
        self.cnn1_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Xavier Initialization
        nn.init.xavier_uniform_(self.cnn1_1.weight)
        
        self.cnn1_2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Batch Normalization
        self.cnnBN1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        
        self.dropout = nn.Dropout(p=0.8)

        # Convolution Layer 2
        self.cnn2_1 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnn2_2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.cnnBN2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        self.dropout = nn.Dropout(p=0.8)

        # Convolution Layer 3
        self.cnn3_1 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.cnn3_2 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.cnn3_3 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.cnnBN3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        
        # Max Pooling 3
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        
        # Dropout Regularization
        self.dropout = nn.Dropout(p=0.8)

        # Convolution Layer 4
        self.cnn4_1 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.cnn4_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.cnn4_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.cnnBN4 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Max Pooling 4
        self.maxpool4 = nn.MaxPool2d(kernel_size=2)
        
        # Dropout Regularization
        self.dropout = nn.Dropout(p=0.5)

        # Convolution Layer 5
        self.cnn5_1 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.cnn5_2 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.cnn5_3 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.cnnBN5 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)

        # Max Pooling 5
        self.maxpool5 = nn.MaxPool2d(kernel_size=2)
        
        # Dropout Regularization
        self.dropout = nn.Dropout(p=0.8)
#------------------------------------------------
        # Fully connected linear layer
        #self.fc1 = nn.Linear(32*75*75 , 9)  #32 channels, 75x75 final image size
        self.fc1 = nn.Linear(512*image_size*image_size, 4096)  #32 channels, 7x7 final image size
        self.relu4 = nn.ReLU()
        
        self.fc2 = nn.Linear(4096, 1000)  #32 channels, 7x7 final image size
        self.relu5 = nn.ReLU()
        
        self.fc3 = nn.Linear(1000, 10)  #32 channels, 7x7 final image size
        
        
	
	#Image size = 28x28 -> 13x13 after first pooling
	#14x14 after padding = 1
	#7x7 after second pooling

    def forward(self, x):
        """ Forward Propogation for classification """
        
        #CNN layer 1
        out = self.cnn1_1(x)
        out = F.relu(out)
        out = self.cnn1_2(out)
        out = F.relu(out)
        out = self.cnnBN1(out)
        out = self.maxpool1(out)

        #out = self.dropout(out)
        
        #CNN layer 2
        out = self.cnn2_1(out)
        out = F.relu(out)
        out = self.cnn2_2(out)
        out = F.relu(out)
        out = self.cnnBN2(out)
        out = self.maxpool2(out)
        
        #out = self.dropout(out)
        
        #CNN layer 3
        out = self.cnn3_1(out)
        out = F.relu(out)
        out = self.cnn3_2(out)
        out = F.relu(out)
        out = self.cnn3_3(out)
        out = F.relu(out)
        out = self.cnnBN3(out)
        out = self.maxpool3(out)
        
        #out = self.dropout(out)
        
        #CNN layer 4
        out = self.cnn4_1(out)
        out = F.relu(out)
        out = self.cnn4_2(out)
        out = F.relu(out)
        out = self.cnn4_3(out)
        out = F.relu(out)
        out = self.cnnBN4(out)
        out = self.maxpool4(out)
        
        #out = self.dropout(out)
        
        #CNN layer 5
        out = self.cnn5_1(out)
        out = F.relu(out)
        out = self.cnn5_2(out)
        out = F.relu(out)
        out = self.cnn5_3(out)
        out = F.relu(out)
        out = self.cnnBN5(out)
        out = self.maxpool5(out)
        
        out = self.dropout(out)
        
        # Resize the tensor, -1 decides the best dimension automatically
        #out = out.view(out.size(0), -1)
        out = out.view(out.size(0), -1)

        # Dropout
        #out = self.dropout(out)

        # Fully connected 1
        out = self.fc1(out)
        out = F.relu(out)

        out = self.fc2(out)
        out = F.relu(out)

        out = self.fc3(out)
       
        out = F.log_softmax(out, dim=0)  #Softmax along Row
        # Return
        return out
