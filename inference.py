import utils
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
import sympy as sm
sm.init_printing()

# module to save results
import joblib as jb

# pytorch
import torch
import torch.nn as nn

# split data into a training set and a test set
from sklearn.model_selection import train_test_split
# linearly transform a feature to zero mean and unit variance
from sklearn.preprocessing import StandardScaler

# to reload modules
import importlib
import mplhep as hep
hep.style.use("CMS") # string aliases work too
import run_training_consecutive as run_training

import argparse

parser=argparse.ArgumentParser(description='train for different targets')
parser.add_argument('--T', type=str, help='the target that you want. Options: [RecoDatapT, RecoDataeta, RecoDataphi, RecoDatam]', required=True)
args = parser.parse_args()
#target string
T = args.T
target = T
source  = FIELDS[target]
features= source['inputs']


data    = pd.read_csv('Data.csv')
print('number of entries:', len(data))

columns = list(data.columns)[1:]
print('\nColumns:', columns)
print()

fields  = list(data.columns)[5:]
data    = data[fields]

X       = ['genDatapT', 'genDataeta', 'genDataphi', 'genDatam', 'tau']

FIELDS  = {'RecoDatapT' : {'inputs': X, 
                           'xlabel': r'$p_T$ (GeV)', 
                           'xmin': 0, 
                           'xmax':80},
           
           'RecoDataeta': {'inputs': ['RecoDatapT']+X, 
                           'xlabel': r'$\eta$', 
                           'xmin'  : -8, 
                           'xmax'  :  8},
           
           'RecoDataphi': {'inputs': ['RecoDatapT','RecoDataeta']+X, 
                           'xlabel': r'$\phi$',
                           'xmin'  : -4,
                           'xmax'  :  4},
           
           'RecoDatam'  : {'inputs': ['RecoDatapT',
                                      'RecoDataeta','RecoDataphi']+X,
                           'xlabel': r'$m$ (GeV)',
                           'xmin'  : 0, 
                           'xmax'  :20}
          }






#######################RUN/TEST/VALID DATA#################################


# Fraction of the data assigned as test data
fraction = 20/100
# Split data into a part for training and a part for testing
train_data, test_data = train_test_split(data, 
                                         test_size=fraction)

# Split the training data into a part for training (fitting) and
# a part for validating the training.
fraction = 5/80
train_data, valid_data = train_test_split(train_data, 
                                          test_size=fraction)

# reset the indices in the dataframes and drop the old ones
train_data = train_data.reset_index(drop=True)
valid_data = valid_data.reset_index(drop=True)
test_data  = test_data.reset_index(drop=True)

print('train set size:        %6d' % train_data.shape[0])
print('validation set size:   %6d' % valid_data.shape[0])
print('test set size:         %6d' % test_data.shape[0])

# create a scaler for target
scaler_t = StandardScaler()
scaler_t.fit(train_data[target].to_numpy().reshape(-1, 1))

# create a scaler for inputs
scaler_x = StandardScaler()
scaler_x.fit(train_data[features])

# NB: undo scaling of tau, which is always the last feature
#this is a nice trick!
scaler_x.mean_[-1] = 0
scaler_x.scale_[-1]= 1

scalers = [scaler_t, scaler_x]

train_t, train_x =utils. split_t_x(train_data, target, features, scalers)
valid_t, valid_x = utils.split_t_x(valid_data, target, features, scalers)
test_t,  test_x  = utils.split_t_x(test_data,  target, features, scalers)


print('TARGETS ARE', train_t)
print()
print('TRAINING FEATURES', train_x)

print(train_t.shape, train_x.shape)




#############

import torch.nn as nn

dnn = nn.Sequential(
                    nn.Linear( train_x.shape[1], 50),
                      nn.ReLU(),
                      
                      nn.Linear(50, 50),
                      nn.ReLU(),
                      
                      nn.Linear(50, 50),
                      nn.ReLU(), 
 
                      nn.Linear(50, 50),
                      nn.ReLU(), 
 
                      nn.Linear(50, 1))

dnn.load_state_dict(torch.load('trained_models/iqn_model_CONSECUTIVE_%s.dict' % target))

if T== 'RecoDatapT':
    label= '$p_T$ [GeV]'
    x_min, x_max = 20, 60
elif T== 'RecoDataeta':
    label = '$\eta$'
    x_min, x_max = -5.4, 5.4
elif T =='RecoDataphi':
    label='$\phi$'
    x_min, x_max = -3.4, 3.4
elif T == 'RecoDatam':
    label = ' $m$ [GeV]'
    x_min, x_max = 0, 18

# traces = ([], [], [], [])
# dnn = run_training.run(model, scalers, target, 
#           train_x, train_t, 
#           valid_x, valid_t, traces)

# run_training.plot_model(test_data, dnn)
