# Importing libraries
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Hyperparameter initialization
n_epoch         = 6
n_class         = 9
batch_size      = 1
learning_rate   = 0.001

# check if GPU is available
print(torch.cuda.current_device())
print(torch.cuda.device(0))
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

#To run on GPU
device = torch.device("cuda:0")

# Sorting out the data

# Image parameters
img_size = 300
train_img = "../TrainingData"
test_img = "../TestData"


# Define the transformation
transform = transforms.Compose( [#transforms.Resize(img_size),
                                 transforms.CenterCrop(img_size),
                                 transforms.Grayscale(num_output_channels=1),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
                                 ])

# Training dataset
train_dataset = datasets.ImageFolder(root=train_img, transform=transform)

# Testing dataset
test_dataset = datasets.ImageFolder(root=test_img, transform=transform)

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
        #self.fc1 = nn.Linear(32*75*75 , 9)  #32 channels, 75x75 final image size
        self.fc1 = nn.Linear(32*75*75 , 9)  #32 channels, 7x7 final image size
	
	#Image size = 28x28 -> 13x13 after first pooling
	#14x14 after padding = 1
	#7x7 after second pooling

    def forward(self, x):
        """ Forward Propogation for classification """
        
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
#        print("size of out1:", out.shape)

        # Max pool 1
        out = self.maxpool1(out)
#        print("size of out1 maxpool:", out.shape)

        # Convolution 2
        out = self.cnn2(out)
        out = self.relu2(out)
#        print("size of out2:", out.shape)

        # Max pool 2
        out = self.maxpool2(out)
#        print("size of out2 maxpool:", out.shape)
        
        # Resize the tensor, -1 decides the best dimension automatically
        #out = out.view(out.size(0), -1)
        out = out.view(out.size(0), -1)
#        print("size of out resize:", out.shape)

        # Dropout
        out = self.dropout(out)
#        print("size of out dropout:", out.shape)

        # Fully connected 1
        out = self.fc1(out)
#        print("size of out fc1:", out.shape)
        
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
    total = 0
    correct = 0
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
        total += labels.size(0)

        # obtain prediction from max value
        _, predicted = torch.max(outputs.data, 1)

        # Calculate the number of right answers
        correct += (predicted == labels).sum().item()

        # Prit loss and accuracy
        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}, correct = {} | total = {}'.format(epoch + 1, n_epoch, i + 1, len(train_loader), loss.item(), (correct / total) * 100, correct, total))


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
        print("predicted: {} | Correct: {} %".format(predicted, labels))

print('Test Accuracy of the model on the {} test images: {} %'.format(len(test_loader), 100 * correct / total))

