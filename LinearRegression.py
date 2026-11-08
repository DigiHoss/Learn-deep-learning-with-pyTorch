import torch 
import torch.nn as nn
import numpy as np 
from sklearn import datasets
import matplotlib.pyplot as plt 

# step 0: prepare our data 
## generating a random dataset that follows a regresssion model which contains :
### 100 data samples 
### 1 feature (a varibale (attribure) that represents a specific information))
X_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=1)

## take a look on those arrays shapes and dimensions (dim)
print(f"Data array shape: {X_numpy.shape} ; dimension: {X_numpy.ndim}")
print(f"Data array shape: {y_numpy.shape} ; dimension: {y_numpy.ndim}")

## convert a numpy array to a torch tensor 
X = torch.from_numpy(X_numpy.astype(np.float32))
y = torch.from_numpy(y_numpy.astype(np.float32))

### Based on the shapes and dimensions, X is gonna be good but y should be reshaped 
### but why ?, take a look to the following example 

# ======================= Example of Data ==========================
"""
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])  # Feature input
y = torch.tensor([[2.0], [4.0], [6.0], [8.0]])  # Target output
"""
# ======================= End of Example ==========================

# in the example, Y has the same shape of X; 
### Let's do it in our case
y = y.view(y.shape[0], 1)

# step 1: Design the model 

n_samples, n_features = X.shape
input_size = n_features
output_size = 1 

## define a simple linear model: y = wx + b

model = nn.Linear(in_features=input_size, out_features=output_size)

# step 2: Construct loss and optimizer

## define the loss, here the Mean Squared Error loss function (MSE)
criterion = nn.MSELoss()

## define the optimizer, here we use Stochastic Gradient Descent Optimizer (SGD)
learning_rate = 0.01 # the step 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


# step 3: Training loop 
num_epochs = 100
for epoch in range(num_epochs):
    # forwared pass : Predict output and compute loss
    y_predicted = model(X)
    loss = criterion(y_predicted, y)
    
    # backward pass : compute gradients 
    loss.backward()
    
    # update weights using gradients
    optimizer.step()
    
    # clear previous gradients 
    optimizer.zero_grad() 
    
    if (epoch+1) % 10 == 0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4f}")
        
# plot 
predicted = model(X).detach().numpy()
plt.plot(X.numpy(), y.numpy(), 'ro')
plt.plot(X.numpy(), predicted, 'b')
plt.show()