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

# Hyperparameter initialization
n_epoch         = 1 
n_class         = 10
batch_size      = 1 
learning_rate   = 0.0001

# check if GPU is available
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

#To run on GPU
device = torch.device("cuda:0")
dtype = torch.float
# Sorting out the data

# Image parameters
n_cnn = 3 #Number of CNN layer
img_size = (256,256)
conv_size = int( img_size[0]/ (2**(n_cnn+1)) ) # image_size / 8 for 3 cnn layer. i.e 2**3 = 8
train_img = "../TrainingData"
Model = "./Model"

# Define the transformation
transform = transforms.Compose( [transforms.Resize(img_size),
                                 #transforms.Grayscale(num_output_channels=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                 #transforms.Normalize((0.5),(0.5))
                                 ])

# Training dataset
train_dataset = datasets.ImageFolder(root=train_img, transform=transform)

# Placing data into dataloader for better accessibility
# Shuffle training dataset to eleminate bias
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
  
# Instance creation
#model = CNNModel(conv_size, n_class).cuda()
model = nn.DataParallel(CNNModel(conv_size, n_class))
model = model.to(device)
# Create instance of loss
criterion = nn.CrossEntropyLoss()

# Create instance of optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model Training

n_iteration = 0

for epoch in range(n_epoch):
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(train_loader):
        # Wrap into Variable
        #images = Variable(images).cuda()
        #labels = Variable(labels).cuda()
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward propogation
        #outputs = model(images).cuda()
        outputs = model(images)

        # Loss calculation ( softmax )
        loss = criterion(outputs, labels)

        # Backpropogation
        loss.backward()

        # Update Gradients
        optimizer.step()
        
        n_iteration += 1

        # Total number of labels
        total += labels.size(0)

        # obtain prediction from max value
        _, predicted = torch.max(outputs.data, 1)

        # Calculate the number of right answers
        correct += (predicted == labels).sum().item()

        # Prit loss and accuracy
        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%, correct = {} | total = {}'.format(epoch + 1, n_epoch, i + 1, len(train_loader), loss.item(), (correct / total) * 100, correct, total))

# Saving the trained model            
if not os.path.exists(Model):
    os.makedirs(Model)
torch.save(model.state_dict(), "./Model/model.pth")
print("Model saved at ./Model/model.pth")
