import torch
import torch.nn as nn 
import numpy as np 
from sklearn import datasets
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split 

# step 0: prepare data 
bc = datasets.load_breast_cancer() # a dataset that is already available on scikit-learn

## take a look on some details about this dataset 
## what should I do when facing every new dataset ? 
keys = bc.keys() # first this function return what available in this dataset 
# print(keys)
"""
dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])
"""

## so 'data', 'target' are common on every dataset, but you should absolutely make sure understanding 'feature names'
# print(bc['feature_names'])

## then, take a look to 'target_names'
# print(bc['target_names'])

X, y = bc.data, bc.target 

## then, you should know the size of your data and the number of features you've seen just before 
n_samples, n_features = X.shape
print(n_samples, n_features)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

## scale our data : very recommended when using Logistic regression 
### it's basically make our features to have zero mean and unit variance 


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

y_train = y_train.view(y_train.shape[0], 1)
y_test = y_test.view(y_test.shape[0], 1)


# step 1: Design the model

class LogisticRegression(nn.Module): 
    def __init__(self, n_input_features):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        
    def forward(self, x):
        y_predicted = torch.sigmoid(self.linear(x)) # apply sigmoid to output
        return y_predicted
    

model = LogisticRegression(n_features) # instantiate the model

# step 2: loss and optimizer 
learning_rate = 0.01
criterion = nn.BCELoss() # Binary Cross-Entropy loss for binary classification
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    
# step 3: training loop 
num_epochs = 100
for epoch in range(num_epochs):
    # forward pass 
    y_predicted = model(X_train)
    loss = criterion(y_predicted, y_train)
    
    # backward pass 
    loss.backward()
    
    # updates 
    optimizer.step()
    
    # zero gradients 
    optimizer.zero_grad()
    
    if (epoch+1)%10==0:
        print(f"epoch: {epoch+1}, loss = {loss.item():.4}")   
    
with torch.no_grad():
    y_predicted = model(X_test)
    y_predicted_cls = y_predicted.round()
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f"accuracy = {acc:.4f}")
    # Print the learned parameters
    params = list(model.parameters())
    print(f'Learned weight: {params[0]}, bias: {params[1].item():.4f}')