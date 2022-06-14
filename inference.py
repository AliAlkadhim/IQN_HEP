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
FONTSIZE = 18
font = {'family' : 'serif',
        'weight' : 'normal',
        'size'   : FONTSIZE}
mp.rc('font', **font)
mp.rc('text', usetex=True)

import aliutils as utils
import mplhep as hep
hep.style.use("CMS") # string aliases work too


##########################
import argparse

parser=argparse.ArgumentParser(description='train for different targets')
parser.add_argument('--T', type=str, help='the target that you want. Options: [RecoDatapT, RecoDataeta, RecoDataphi, RecoDatam]', required=True)
args = parser.parse_args()
#target string
T = args.T
target = T

data    = pd.read_csv('Data.csv')
print('number of entries:', len(data))

columns = list(data.columns)[1:]
print('\nColumns:', columns)

fields  = list(data.columns)[5:]
print('\nFields:', fields)

# target  = 'RecoDatapT'
print('\nTarget:', target )

features= [x for x in fields]
features.remove(target)

print('\nFeatures:', features)

data    = data[fields]
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

scaler_t = StandardScaler()
scaler_t.fit(train_data[target].to_numpy().reshape(-1, 1))

# create a scaler for inputs
scaler_x = StandardScaler()
scaler_x.fit(train_data[features])
# NB: undo scaling of tau, which is the last feature
scaler_x.mean_[-1] = 0
scaler_x.scale_[-1]= 1

scalers = [scaler_t, scaler_x]

train_t, train_x = utils.split_t_x(train_data, target, features, scalers)
valid_t, valid_x = utils.split_t_x(valid_data, target, features, scalers)
test_t,  test_x  = utils.split_t_x(test_data,  target, features, scalers)

print(train_t.shape, train_x.shape)



model =  utils.RegressionModel(nfeatures=train_x.shape[1], 
               ntargets=1,
               nlayers=8, 
               hidden_size=4, 
               dropout=0.3)



PATH='trained_models/IQN_100kRecoDatapT.dict'
# 'trained_models/IQN_100k'+T+'.dict
model.load_state_dict(torch.load(PATH))
print(model)

# T='RecoDatapT'
if T== 'RecoDatapT':
    label= '$p_T$ [GeV]'
elif T== 'RecoDataeta':
    label = '$\eta$'
elif T =='RecoDataphi':
    label='$\phi$'
elif T == 'RecoDatam':
    label = ' $m$ [GeV]'


y_label_dict ={'RecoDatapT':'$p(p_T)$'+' [ GeV'+'$^{-1} $'+']',
                    'RecoDataeta':'$p(\eta)$', 'RecoDataphi':'$p(\phi)$',
                    'RecoDatam':'$p(m)$'+' [ GeV'+'$^{-1} $'+']'}

def plot_model(df, dnn,
               gfile='fig_model.png', 
               save_image=False,
               fgsize=(8, 8), 
               ftsize=20):
        
    # ----------------------------------------------
    # histogram RecoDatapT
    # ----------------------------------------------
    xmin, xmax = 20, 60
    xbins = 80
    xstep = (xmax - xmin)/xbins

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=fgsize)
    
    ax.set_xlim(xmin, xmax)

    #ax.set_ylim(ymin, ymax)
    # ax.set_xlabel(r'$p_{T}$ (GeV)', fontsize=ftsize)
    ax.set_xlabel('reco jet '+label, fontsize=ftsize)
    ax.set_ylabel(y_label_dict[T], fontsize=ftsize)

    ax.hist(df.RecoDatapT, 
            bins=xbins, 
            range=(xmin, xmax), alpha=0.4, color='blue',
            label='Data')
   
    y = dnn(df)
    
    ax.hist(y, 
            bins=xbins, 
            range=(xmin, xmax), 
            alpha=0.4, 
            color='red', label='IQN moodel')
    ax.grid()

    plt.tight_layout()
    plt.legend()
    if save_image:
        plt.savefig(gfile)
    plt.show()

dnn = utils.ModelHandler(model, scalers)


plot_model(test_data, dnn)
