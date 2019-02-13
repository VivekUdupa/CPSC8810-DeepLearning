# Importing libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Hyperparameter initialization
n_epoch         = 20
n_class         = 10
batch_size      = 1
learning_rate   = 1e-6


# Sorting out the data

# Image parameters
img_size = 28
img_dir = "../TrainingData"


# Define the transformation
transform = transforms.Compose( [transforms.Resize(img_size),
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                 ])

# Training dataset
train_dataset = datasets.ImageFolder(root=img_dir, transform=transform)

# Testing dataset
test_dataset = datasets.ImageFolder(root=img_dir, transform=transform)

# Placing data into dataloader for better accessibility
# Shuffle training dataset to eleminate bias
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

#Defining the CNN
class CNNModel(nn.Module):
    """ A CNN Model for image classification """
    
    def __init__(self):
        """ CNN layer to process the image"""
        super(CNNModel, self).__init__() # Super is used to refer to the base class, i.e nn.Module

        # Convolution Layer 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        # Max Pooling 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)

        # Convolution Layer 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # Max Pooling 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)

        # Dropout Regularization
        self.dropout = nn.Dropout(p=0.4)

        # Fully connected linear layer
        self.fc1 = nn.Linear(32*7*7, 9)  #32 channels, 7x7 final image size
	
	#Image size = 28x28 -> 13x13 after first pooling
	#14x14 after padding = 1
	#7x7 after second pooling

    def forward(self, x):
        """ Forward Propogation for classification """
        
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)

        # Max pool 1
        out = self.maxpool1(out)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)

        # Max pool 2
        out = self.maxpool2(out)

        # Resize the tensor, -1 decides the best dimension automatically
        out = out.view(out.size(0), -1)

        # Dropout
        out = self.dropout(out)

        # Fully connected 1
        out = self.fc1(out)
        
        # Return
        return out
    
# Instance creation
model = CNNModel()

# Create instance of loss
criterion = nn.CrossEntropyLoss()

# Create instance of optimizer (Adam)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Model Training

n_iteration = 0

for epoch in range(n_epoch):
    for i, (images, labels) in enumerate(train_loader):
        # Wrap into Variable
        images = Variable(images)
        labels = Variable(labels)

        # Clear the gradients
        optimizer.zero_grad()

        # Forward propogation
        outputs = model(images)

        # Loss calculation ( softmax )
        loss = criterion(outputs, labels)

        # Backpropogation
        loss.backward()

        # Update Gradients
        optimizer.step()
        
        n_iteration += 1

        # Total number of labels
        total = labels.size(0)

        # obtain prediction from max value
        _, predicted = torch.max(outputs.data, 1)

        # Calculate the number of right answers
        correct = (predicted == labels).sum().item()

        # Prit loss and accuracy
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'.format(epoch + 1, num_epochs, i + 1, len(train_loader), loss.item(), (correct / total) * 100))


# Testing the model
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = Variable(images)
        labels = Variable(labels)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Test Accuracy of the model on the 10000 test images: {} %'.format(100 * correct / total))

