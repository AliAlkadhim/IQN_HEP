import os, sys

# the standard module for tabular data
import pandas as pd

# the standard module for array manipulation
import numpy as np

# the standard modules for high-quality plots
import matplotlib as mp
import matplotlib.pyplot as plt

# standard scientific python module
import scipy as sp
import scipy.stats as st
import scipy.optimize as op

# standard symbolic algebra module
#import sympy as sm
#sm.init_printing()

# module to save results
import joblib as jb

# pytorch
import torch
import torch.nn as nn
from torch.utils import data

# split data into a training set and a test set
from sklearn.model_selection import train_test_split
# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

# to reload modules
import importlib

get_ipython().run_line_magic("matplotlib", " inline")

import matplotlib as mp
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

# update fonts
FONTSIZE = 18
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)

# set usetex = False if LaTex is not 
# available on your system or if the 
# rendering is too slow
mp.rc('text', usetex=True)

# set a seed to ensure reproducibility
#seed = 128
#rnd  = np.random.RandomState(seed)

from sklearn.model_selection import train_test_split
import copy


df = pd.read_csv('Data.csv')
df.head()


df = df.iloc[:,5:]
df.head()


levels = ['genData', 'RecoData']
kinematics=['pT','eta','phi','m']
targets = kinematics#for reco level, but same names
Networks = ['RecoNN', 'genNN']

target = df['RecoDatapT'].to_numpy()
data =  df.drop('RecoDatapT', axis=1).to_numpy()
data


target


# train_targets = train_targets.reshape(-1,1)
# test_targets = test_targets.reshape(-1,1)

print('target shape', train_targets.shape)
print('input data shape', data.shape)


ntargets = 1
train_data, test_data, train_targets, test_targets = train_test_split(data, target, test_size=0.2)



train_targets = train_targets.reshape(-1,1)
test_targets = test_targets.reshape(-1,1)

sets= [train_data, test_data, train_targets, test_targets]
set_names = ['train_data', 'test_data', 'train_targets', 'test_targets']
# vnames = [name for name in globals() if globals()[name] is variable]

def variable_string(variable):
    return [k for k, v in locals().items() if v == variable][0]

for var_name, var in zip(set_names, sets):
    print(var_name 
          + ' shape = ', var.shape, '\n')


# sc = StandardScaler()#this is always recommended for logistic regression
# train_data= sc.fit_transform(train_data)
# test_data = sc.transform(test_data)
# train_data.mean(), (train_data.std())**2#check to make sure mean=0, std=1


class CustomDataset:
    """This takes the index for the data and target and gives dictionary of tensors of data and targets.
    For example we could do train_dataset = CustomDataset(train_data, train_targets); test_dataset = CustomDataset(test_data, test_targets)
 where train and test_dataset are np arrays that are reshaped to (-1,1).
 Then train_dataset[0] gives a dictionary of samples "X" and targets"""
    def __init__(self, data, targets):
        self.data = data
        self.targets=targets
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, idx):
        
        current_sample = self.data[idx, :]
        current_target = self.targets[idx]
        return {"x": torch.tensor(current_sample, dtype = torch.float),
               "y": torch.tensor(current_target, dtype= torch.float),
               }#this already makes the targets made of one tensor (of one value) each
    
train_dataset = CustomDataset(train_data, train_targets)
test_dataset = CustomDataset(test_data, test_targets)
print(train_dataset[0], train_dataset)


batch_size=5
train_loader = torch.utils.data.DataLoader(train_dataset, 
                                           batch_size=batch_size, 
                                           num_workers=2, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(test_dataset, 
                                          batch_size=batch_size, num_workers=2)


# from mymodels import RegressionModel
class RegressionModel(nn.Module):
    #inherit from the super class
    def __init__(self, nfeatures, ntargets, nlayers, hidden_size, dropout):
        super().__init__()
        layers = []
        for _ in range(nlayers):
            if len(layers) ==0:
                #inital layer has to have size of input features as its input layer
                #its output layer can have any size but it must match the size of the input layer of the next linear layer
                #here we choose its output layer as the hidden size (fully connected)
                layers.append(nn.Linear(nfeatures, hidden_size))
                #batch normalization
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                #ReLU activation 
                layers.append(nn.ReLU())
            else:
                #if this is not the first layer (we dont have layers)
                layers.append(nn.Linear(hidden_size, hidden_size))
                layers.append(nn.BatchNorm1d(hidden_size))
                layers.append(nn.Dropout(dropout))
                layers.append(nn.ReLU())
                #output layer:
        layers.append(nn.Linear(hidden_size, ntargets)) 
        
        layers.append(nn.Sigmoid())
            #we have defined sequential model using the layers in oulist 
        self.model = nn.Sequential(*layers)
            
    
    def forward(self, x):
        return self.model(x)


# n_examples, n_inputs = train_data.shape
# n_outputs, n_hidden = 1, 16
print('train_data.shape = ',train_data.shape)


model =  RegressionModel(nfeatures=train_data.shape[1], 
               ntargets=1,
               nlayers=8, 
               hidden_size=16, 
               dropout=0.3)
print(model)


# get_ipython().run_line_magic("writefile", " training/RegressionEngine.py")
class RegressionEngine:
    """loss, training and evaluation"""
    def __init__(self, model, optimizer):
                 #, device):
        self.model = model
        #self.device= device
        self.optimizer = optimizer
        
    #the loss function returns the loss function. It is a static method so it doesn't need self
    @staticmethod
    def quadratic_loss(targets, outputs):
         return nn.MSELoss()(outputs, targets)

    @staticmethod
    def average_quadratic_loss(targets, outputs):
    # f and t must be of the same shape
        return  torch.mean((outputs - targets)**2)
    
    @staticmethod
    def average_cross_entropy_loss(targets, outputs):
        # f and t must be of the same shape
        loss = torch.where(targets > 0.5, torch.log(outputs), torch.log(1 - outputs))
        return -torch.mean(loss)
    
    @staticmethod
    def average_quantile_loss(targets, outputs):
        # f and t must be of the same shape
        tau = torch.rand(outputs.shape)
        return torch.mean(torch.where(targets >= outputs, 
                                      tau * (targets - outputs), 
                                      (1 - tau)*(outputs - targets)))

    def train(self, data_loader):
        """the training function: takes the training dataloader"""
        self.model.train()
        final_loss = 0
        for data in data_loader:
            self.optimizer.zero_grad()#only optimize weights for the current batch, otherwise it's meaningless!
            inputs = data["x"]
            targets = data["y"]
            outputs = self.model(inputs)
            loss = self.average_quadratic_loss(targets, outputs)
            loss.backward()
            self.optimizer.step()
            final_loss += loss.item()
            return final_loss / len(data_loader)

    
    def evaluate(self, data_loader):
        """the training function: takes the training dataloader"""
        self.model.eval()
        final_loss = 0
        for data in data_loader:
            inputs = data["x"]#.to(self.device)
            targets = data["y"]#.to(self.device)
            outputs = self.model(inputs)
            loss = self.average_quadratic_loss(targets, outputs)
            final_loss += loss.item()
            return outputs.flatten()
            #return final_loss / len(data_loader)


def train(optimizer, engine, early_stopping_iter, epochs):
    train_losses, test_losses = [], []
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    eng = RegressionEngine(model=model, optimizer = optimizer)
    best_loss = np.inf
    early_stopping_iter = 10
    early_stopping_counter = 0
    # EPOCHS=22
    EPOCHS=epochs
    for epoch in range(EPOCHS):
        train_loss = eng.train(train_loader)
        test_loss = eng.train(test_loader)
        print("Epoch : get_ipython().run_line_magic("-10g,", " Training Loss: %-10g, Test Loss: %-10g\" % (epoch, train_loss, test_loss))")
        #print(f"{epoch}, {train_loss}, {test_loss}")
        if test_loss < best_loss:
            best_loss = test_loss

        else:
            early_stopping_counter += 1

        if early_stopping_counter > early_stopping_iter:
            #if we are not improving for 10 iterations then break the loop
            #we could save best model here
            break
    
        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    train_losses=np.array(train_losses); test_losses=np.array(test_losses)
    
    fig = plt.figure(figsize=(5, 5))
    
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)
    ax.set_title("Average loss")
    
    epoch_list = np.arange(1, train_losses.shape[0]+1)
    ax.plot(epoch_list, train_losses, label = 'Train')
    ax.plot(epoch_list, test_losses, label='Test')
    ax.set_xlabel('Epoch')
    ax.legend(loc='upper right')
    return train_losses, test_losses

    


optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
train_losses, test_losses=train(optimizer, 
      engine =RegressionEngine(model=model, optimizer = optimizer),
      early_stopping_iter = 100,
      epochs=100)


np.array(train_losses).shape, np.array(test_losses).shape


# plt.plot(np.arange(1, train_losses.shape


def predict():
    outputs = []
    labels = []
    accuracies = []

    #evaluate
    with torch.no_grad():
        for data in test_loader:
            data_cp = copy.deepcopy(data)

            xtest = data_cp["x"]
            ytest = data_cp["y"]#y is Z values. I could add here my computed p-value for each theta,
            #and make a dataframe col1:theta, col2: Z, col3, phat, col4: computedp-value
            output = model(xtest)
            labels.append(ytest)
            outputs.append(output)

            y_predicted_cls = output.round()
            acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])# number of correct predictions/sizeofytest
            #accuracies.append(acc.numpy())
            #print(f'accuracy: {acc.item():.4f}')

            del data_cp

    #     acc = y_predicted_cls.eq(ytest).sum() / float(ytest.shape[0])
    #     print(f'accuracy: {acc.item():.4f}')
            
    OUTPUTS = torch.cat(outputs).view(-1).numpy()

    LABELS = torch.cat(labels).view(-1).numpy()
    print('outputs of model: ', OUTPUTS)
    print('\nactual labels (targets Z): ', LABELS)
    return OUTPUTS.flatten(), LABELS.flatten()


OUTPUTS, LABELS = predict()


plt.hist(LABELS)


def calc_phat_from_regressor(model, test_data):
    X_torch = torch.from_numpy(X).float()
    X_torch= Tensor(X_torch)
    model.eval()
    phat = model(X_torch)
    phat = phat.squeeze()
    phat=phat.detach().numpy().flatten()#detaches it from the computational history/prevent future computations from being tracked
    
