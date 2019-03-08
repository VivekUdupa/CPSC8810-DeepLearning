 A deep learning algorithm to classify images with cyberbully actions. 

Project Team:
1. Shashi Kumar Honnahalli Shivaraju (shonnah@clemson.edu)
2. Vivek Koodli Udupa (vkoodli@clemson.edu)

====================================================
The CNN network structure is as follows:
====================================================
1) Image Pre-Processing: Given input jpeg/jpg image is converted into a mono-channel 256 x 256 image. Then the values are converted into PyTorch Tensors and normalised.
2) CNN Layer 1( in_channel = 1, out_channel = 16, stride = 1, padding = 1, kernel = 3x3)
3) ReLU activation layer
4) Max Pooling layer 1( kernel = 2x2)
5) CNN Layer 2( in_channel = 16, out_channel = 32, stride = 1, padding = 1, kernel = 3x3)
6) ReLU activation layer
7) Max Pooling layer 2( kernel = 2x2)
8) CNN Layer 3( in_channel = 32, out_channel = 32, stride = 1, padding = 1, kernel = 3x3)
9) ReLU activation layer
10) Max Pooling layer 3( kernel = 2x2)
11) Flattening Layer ( Converts 2D feature map to 1D feature map)
12) Dropout
13) Fully Connected Layer 1 with ReLU ( Maps 1D feature map to 5000 neurons)
14) Fully Connected Layer 2 with ReLU ( Maps 5000 neurons to 500 neurons)
15) Fully Connected Layer 3 with ReLU ( Maps 500 neurons to 250 neurons)
16) Fully Connected Layer 4 with ReLU ( Maps 250 neurons to 100 neurons)
17) Fully Connected Layer 5 with ReLU ( Maps 100 neurons to 10 output neurons)
18) Softmax
19) Output

====================================================
Testing:
====================================================
test.py takes a single image file as argument. The image file must be in the jpeg/jpg format. Currently it is the only supported format.

execution example:

$python test.py an_image.jpg

This code will load the model named 'model.pth' from the 'Model' folder. Please make sure to have the trianed model named as 'model.pth' in a folder named 'Model' and place the folder in the same location as the test.py script. 

====================================================
File Descriptions:
====================================================
File name			Description
----------------------------------------------------
detection_Model.py		Python code of the CNN model described above.
imageView.py			Python code to display a jpeg/jpg image
training.py			Python code to train the CNN model defined in the detection_Model.py code.
test.py				Python code to test the trained model
README.txt			This file
REPORT/				This folder consists of the Midterm Report.
PyTorch_Exercise/		This folder consists of practice codes used while getting used to working with PyTorch 

