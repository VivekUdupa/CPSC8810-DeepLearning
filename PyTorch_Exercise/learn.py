import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
from skimage import io, transform
from PIL import Image

#To randomize every execution
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)

#Image size
image_size = 256

#Dataset Directory
img_folder = "../../data"

#Miltiple folders exist inside 'TrainingData' Directory. I think pytorch takes these folder names as different classes for classification. 

#The compose function allows for multiple transforms
#transforms.ToTensor() converts our PILImage to a tensor of shape (C x H x W) in the range [0,1]
#transforms.Normalize(mean,std) normalizes a tensor to a (mean, std) for (R, G, B)
transform = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Resize(image_size)
    ])

#train_set = torchvision.datasets.ImageFolder(root=img_folder, transform=transform)
train_set = torchvision.datasets.ImageFolder(root=img_folder)

#Load the data into an object
data_loader = torch.utils.data.DataLoader(train_set, batch_size=4, shuffle=True, num_workers=2)

print("Images Loaded")

#TO-DO
# 1) Images are stored in python 'Image' format. How to display this? 
# 2) Verify if all images are loaded

#Trying to display the loaded image
img_name = 'lp1022.jpg'
#print(train_set[0])
#plt.imshow(io.imread(os.path.join('../../TrainingData/pullinghair/lp1022.jpg'), img_name))
img = train_set.__getitem__(416)
print(img)
plt.imshow(img[0])
plt.show()
print(img[0].info)

#plt.imshow(torchvision.transforms.ToPILImage()(train_set[0]))
