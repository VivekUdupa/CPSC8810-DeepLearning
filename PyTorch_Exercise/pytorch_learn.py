import torch

dtype = torch.float
device = torch.device("cuda:0") # To run on GPU
#device = torch.device("cpu") # To run on CPU

#N = Batch Size, D_in = Input dimension, H = Hidden Dimension, D_out = output dimension
N, D_in, H, D_out = 64, 1000, 100, 10

# Create Random input and putput data
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# Randomly initialize weights
w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out,  device=device, dtype=dtype)

#Learning Rate
learning_rate = 1e-6

#iterations
epoch = 500

for i in range(epoch):
    # Forward pass: Compute expected y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # Compute and print loss
    loss = (y_pred - y).pow(2).sum().item() # item() converts single tensor number to standard python number
    print(i, loss)

    # Backdrop
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # Update weights using gradient descent
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2

print("Done");
