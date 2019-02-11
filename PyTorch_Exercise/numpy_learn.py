#Numpy Warmup
#Using numpy to manually fit data

import numpy as np

# N         Batch size
# D_in      Input dimension
# H         Hidden Dimension
# D_out     Output Dimension
N, D_in, H, D_out = 64, 1000, 100, 10

#Create Random input and output data
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

#Randomly initialized Weights
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

#A Very small learning rate
learning_rate = 1e-6

#Number of iterations
epoch = 1000

for t in range(epoch):
    #Forward Pass: Compute Predicted y
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    #Compute and print loss
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    #Backdrop to compute gradients of w1 and w2 wrt loss
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    #Update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print("X[0]: ", x[0,0], "y[0]: ", y[0,0], "y_pred[0]: ", y_pred[0,0]) 
