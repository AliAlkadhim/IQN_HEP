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


df = pd.read_csv('Data.csv')
df.head()


data = df.iloc[:,5:]
data.head()


levels = ['genData', 'RecoData']
kinematics=['pT','eta','phi','m']
targets = kinematics#for reco level, but same names
Networks = ['RecoNN', 'genNN']

ex_target = data['RecoDatapT']
ex_data = data.drop('RecoDatapT', axis=1)
ex_data.head()


# Fraction of the data assigned as test data
fraction = 20/75
# Split data into a part for training and a part for testing
train_data, test_data = train_test_split(data, 
                                         test_size=fraction)

# Split the training data into a part for training (fitting) and
# a part for validating the training.
fraction = 5/55
train_data, valid_data = train_test_split(train_data, 
                                          test_size=fraction)

# reset the indices in the dataframes and drop the old ones
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

print('train set size:        get_ipython().run_line_magic("6d'", " % train_data.shape[0])")
print('validation set size:   get_ipython().run_line_magic("6d'", " % valid_data.shape[0])")
print('test set size:         get_ipython().run_line_magic("6d'", " % test_data.shape[0])")


target   = 'RecoDatapT'
features = list(data.columns)
features.remove(target)
print(features)


# def split_t_x(df, target, source):
#     # change from pandas dataframe format to a numpy 
#     # array of the specified types
#     t = np.array(df[target])
#     x = np.array(df[source])
#     return t, x

# train_t, train_x = split_t_x(train_data, target, features)
# valid_t, valid_x = split_t_x(valid_data, target, features)
# test_t,  test_x  = split_t_x(test_data,  target, features)


def load_data(batch_size):
    return (data.DataLoader(train_data, batch_size, shuffle=True, num_workers=4),
            data.DataLoader(test_data, batch_size, shuffle=True, num_workers=4))


batch_size = 50
train_iter, test_iter = load_data(batch_size)
train_iter


n_examples, n_inputs = train_data.shape
n_outputs, n_hidden = 1, 16


from torch.nn import functional as F

class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Module` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(n_input, n_hidden)  # Hidden layer
        self.out = nn.Linear(n_hidden, n_output)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))




def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights);



