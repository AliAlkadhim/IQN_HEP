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


def split_t_x(df, target, source):
    # change from pandas dataframe format to a numpy 
    # array of the specified types
    t = np.array(df[target])
    x = np.array(df[source])
    return t, x

train_t, train_x = split_t_x(train_data, target, features)
valid_t, valid_x = split_t_x(valid_data, target, features)
test_t,  test_x  = split_t_x(test_data,  target, features)


def get_batch(x, t, batch_size):
    # the numpy function choice(length, number)
    # selects at random "batch_size" integers from 
    # the range [0, length-1] corresponding to the
    # row indices.
    rows    = np.random.choice(len(x), batch_size)
    batch_x = x[rows]
    batch_t = t[rows]
    return (batch_x, batch_t)



def average_quantile_loss(f, t, x):
    # f and t must be of the same shape
    tau = x.T[-1] # last column is tau.
    return torch.mean(torch.where(t >= f, 
                                  tau * (t - f), 
                                  (1 - tau)*(f - t)))


class ModelHandler:
    def __init__(self, model):
        self.model = model

    def __call__(self, x, y, *args):
        
        # convert numpy ndarray to a pytorch tensor
        # list(zip(x,y)) converts two arrays into one with two
        # columns.
        
        if len(args) == 0:
            try:
                inp = list(zip(x, y))
            except:
                inp = list([x, y])
        else:
            a = args[0]
            type_a = type(a)
            if type_a in [type(1), type(1.0)]:
                inp = list(zip(x, y, [a]*len(x)))
                
            elif type_a == type([]):
                n = len(a)
                if n < len(x):
                    a = a * int(len(x) / n)
                    n = len(a)
                inp = list(zip(x[:n], y[:n], a))
                
            else:
                inp = list(zip(x, y, a))

        XY = torch.Tensor(inp)
    
        # go to evaluation mode
        self.model.eval()
    
        # compute,reshape to a 1d array, and convert to a numpy array
        Y  = self.model(XY).view(-1,).detach().numpy()
        if len(Y) == 1:
            return Y[0]
        else:
            return Y
        
    def show(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
                print()
                
def validate(model, avloss, inputs, targets):
    # make sure we set evaluation mode so that any training specific
    # operations are disabled.
    model.eval() # evaluation mode
    
    with torch.no_grad(): # no need to compute gradients wrt. x and t
        x = torch.from_numpy(inputs).float()
        t = torch.from_numpy(targets).float()
        # remember to reshape!
        o = model(x).reshape(t.shape)
    return avloss(o, t, x)


def train(model, optimizer, avloss, getbatch,
          train_x, train_t, 
          valid_x, valid_t,
          batch_size, 
          n_iterations, traces, 
          step=50):
    
    # to keep track of average losses
    xx, yy_t, yy_v = traces
    
    n = len(valid_x)
    
    print('Iteration vs average loss')
    print("get_ipython().run_line_magic("10s\t%10s\t%10s"", " % \")
          ('iteration', 'train-set', 'valid-set'))
    
    for ii in range(n_iterations):

        # set mode to training so that training specific 
        # operations such as dropout are enabled.
        model.train()
        
        # get a random sample (a batch) of data (as numpy arrays)
        batch_x, batch_t = getbatch(train_x, train_t, batch_size)
        
        # convert the numpy arrays batch_x and batch_t to tensor 
        # types. The PyTorch tensor type is the magic that permits 
        # automatic differentiation with respect to parameters. 
        # However, since we do not need to take the derivatives
        # with respect to x and t, we disable this feature
        with torch.no_grad(): # no need to compute gradients 
            # wrt. x and t
            x = torch.from_numpy(batch_x).float()
            t = torch.from_numpy(batch_t).float()      

        # compute the output of the model for the batch of data x
        # Note: outputs is 
        #   of shape (-1, 1), but the tensor targets, t, is
        #   of shape (-1,)
        # In order for the tensor operations with outputs and t
        # to work correctly, it is necessary that they have the
        # same shape. We can do this with the reshape method.
        outputs = model(x).reshape(t.shape)
   
        # compute a noisy approximation to the average loss
        empirical_risk = avloss(outputs, t, x)
        
        # use automatic differentiation to compute a 
        # noisy approximation of the local gradient
        optimizer.zero_grad()       # clear previous gradients
        empirical_risk.backward()   # compute gradients
        
        # finally, advance one step in the direction of steepest 
        # descent, using the noisy local gradient. 
        optimizer.step()            # move one step
        
        if ii % step == 0:
            
            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) 
            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n])

            if len(xx) < 1:
                xx.append(0)
                print("get_ipython().run_line_magic("10d\t%10.6f\t%10.6f"", " % \")
                      (xx[-1], acc_t, acc_v))
            else:
                xx.append(xx[-1] + step)
                print("\rget_ipython().run_line_magic("10d\t%10.6f\t%10.6f"", " % \")
                      (xx[-1], acc_t, acc_v), end='')
                
            yy_t.append(acc_t)
            yy_v.append(acc_v)
    print()      
    return (xx, yy_t, yy_v)



def plot_average_loss(traces, ftsize=18):
    
    xx, yy_t, yy_v = traces
    
    # create an empty figure
    fig = plt.figure(figsize=(5, 5))
    fig.tight_layout()
    
    # add a subplot to it
    nrows, ncols, index = 1,1,1
    ax  = fig.add_subplot(nrows,ncols,index)

    ax.set_title("Average loss")
    
    ax.plot(xx, yy_t, 'b', lw=2, label='Training')
    ax.plot(xx, yy_v, 'r', lw=2, label='Validation')

    ax.set_xlabel('Iterations', fontsize=ftsize)
    ax.set_ylabel('average loss', fontsize=ftsize)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.grid(True, which="both", linestyle='-')
    ax.legend(loc='upper right')

    plt.show()


ntargets=1
nfeatures, nexamples  = train_data.shape[1], n_batch
model = nn.Sequential(nn.Linear( nfeatures, n_batch),
                      nn.ReLU(),
                      
                      nn.Linear(n_batch, n_batch),
                      nn.ReLU(),
                      
                      nn.Linear(n_batch, n_batch),
                      nn.ReLU(), 
                      
                      nn.Linear(n_batch, 1), 
                      nn.Sigmoid()) 


n_batch       = 50
n_iterations  = 50000

learning_rate = 5.e-4
optimizer     = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate) 

traces = ([], [], [])
traces_step = 10

traces = train(model, optimizer, 
                  average_quantile_loss,
                  get_batch,
                  train_x, train_t, 
                  valid_x, valid_t,
                  n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step)






