#!/usr/bin/env python
# coding: utf-8

# ## Implicit Quantile Networks for Event Folding
# Created: June 13, 2022 Ali & Harrison<br>
# 
# ### Introduction 
# 
# 

# In[7]:


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

get_ipython().run_line_magic('matplotlib', 'inline')


# In[8]:


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


# ### Load training data

# In[9]:


data    = pd.read_csv('Data.csv')
print('number of entries:', len(data))

columns = list(data.columns)[1:]
print('\nColumns:', columns)

fields  = list(data.columns)[5:]
print('\nFields:', fields)

target  = 'RecoDatapT'
print('\nTarget:', target )

features= [x for x in fields]
features.remove(target)

print('\nFeatures:', features)

data    = data[fields]
data[:5]


# ### Train, validation, and test sets
# There is some confusion in terminology regarding validation and test samples (or sets). We shall adhere to the defintions given here https://machinelearningmastery.com/difference-test-validation-datasets/):
#    
#   * __Training Dataset__: The sample of data used to fit the model.
#   * __Validation Dataset__: The sample of data used to decide 1) whether the fit is reasonable (e.g., the model has not been overfitted), 2) decide which of several models is the best and 3) tune model hyperparameters.
#   * __Test Dataset__: The sample of data used to provide an unbiased evaluation of a final model fit on the training dataset.
# 
# The validation set will be some small fraction of the training set and can be used, for example, to decide when to stop the training.

# In[10]:


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


# Split data into targets $t$ and inputs $\mathbf{x}$

# In[12]:


def split_t_x(df, target, source, scalers):
    # change from pandas dataframe format to a numpy array
    scaler_t, scaler_x = scalers
    t = np.array(scaler_t.transform(df[target].to_numpy().reshape(-1, 1)))
    x = np.array(scaler_x.transform(df[source]))
    t = t.reshape(-1,)
    return t, x

# create a scaler for target
scaler_t = StandardScaler()
scaler_t.fit(train_data[target].to_numpy().reshape(-1, 1))

# create a scaler for inputs
scaler_x = StandardScaler()
scaler_x.fit(train_data[features])
# NB: undo scaling of tau, which is the last feature
scaler_x.mean_[-1] = 0
scaler_x.scale_[-1]= 1

scalers = [scaler_t, scaler_x]

train_t, train_x = split_t_x(train_data, target, features, scalers)
valid_t, valid_x = split_t_x(valid_data, target, features, scalers)
test_t,  test_x  = split_t_x(test_data,  target, features, scalers)

train_t.shape, train_x.shape


# ### Quantile regression
# 
# The empirical risk, which is the __objective function__ we shall minimize, is defined as
# 
# \begin{align}
# R_M(\theta) & = \frac{1}{M}\sum_{m=1}^M L(t_m, f_m),\\
# \text{where } f_m & \equiv f(\mathbf{x}_m, \theta) .
# \end{align}
# 
# We shall use the __quantile loss__ defined by
# 
# \begin{align}
# L(t, f) & = \begin{cases}
#     \tau (t - f), & \text{if } t \geq f \\
#     (1 - \tau)(f - t)              & \text{otherwise} .
# \end{cases}
# \end{align}
# 
# To show that this loss function indeed leads to $\tau$-quantiles, consider the minimization of the __risk functional__
# 
# \begin{align}
# R[f] & = \int \int \, p(t, \mathbf{x}) \, L(t, f(\mathbf{x}, \theta)) \, dt \, d\mathbf{x}, \\
# \frac{\delta R}{\delta f}  & = 0 .
# \end{align}
# 
# If the above is to hold $\forall\,\,\mathbf{x}$, then 
# 
# \begin{align}
# \int \frac{\partial L}{\partial f} p(t | \mathbf{x}) \, dt & = 0,\\
# -\tau \int_{t \ge f} p(t | \mathbf{x}) \, dt + (1 - \tau)\int_{t < f} p(t | \mathbf{x}) \, dt & = 0, \\
# \therefore \quad\int_{t < f} p(t | \mathbf{x}) \, dt & = \tau.
# \end{align}

# ### Save training utilities to file SIR_dnn_util.py 
# 
#   1. get_batch
#   1. average_quadratic_loss
#   1. average_cross_entropy_loss
#   1. average_quantile_loss
#   1. validate
#   1. ModelHandler
#   1. train
#   1. plot_average_loss
#   1. hist_data

# In[19]:


get_ipython().run_cell_magic('writefile', 'iqnutil.py', '\nimport numpy as np\n\n# the standard modules for high-quality plots\nimport matplotlib as mp\nimport matplotlib.pyplot as plt\n\nimport torch\nimport torch.nn as nn\n\n# return a batch of data for the next step in minimization\ndef get_batch(x, t, batch_size):\n    # the numpy function choice(length, number)\n    # selects at random "batch_size" integers from \n    # the range [0, length-1] corresponding to the\n    # row indices.\n    rows    = np.random.choice(len(x), batch_size)\n    batch_x = x[rows]\n    batch_t = t[rows]\n    return (batch_x, batch_t)\n\n# Note: there are several average loss functions available \n# in pytorch, but it\'s useful to know how to create your own.\ndef average_quadratic_loss(f, t, x):\n    # f and t must be of the same shape\n    return  torch.mean((f - t)**2)\n\ndef average_cross_entropy_loss(f, t, x):\n    # f and t must be of the same shape\n    loss = torch.where(t > 0.5, torch.log(f), torch.log(1 - f))\n    return -torch.mean(loss)\n\ndef average_quantile_loss(f, t, x):\n    # f and t must be of the same shape\n    tau = x.T[-1] # last column is tau.\n    return torch.mean(torch.where(t >= f, \n                                  tau * (t - f), \n                                  (1 - tau)*(f - t)))\n\n# function to validate model during training.\ndef validate(model, avloss, inputs, targets):\n    # make sure we set evaluation mode so that any training specific\n    # operations are disabled.\n    model.eval() # evaluation mode\n    \n    with torch.no_grad(): # no need to compute gradients wrt. x and t\n        x = torch.from_numpy(inputs).float()\n        t = torch.from_numpy(targets).float()\n        # remember to reshape!\n        o = model(x).reshape(t.shape)\n    return avloss(o, t, x)\n\n# A simple wrapper around a model to make using the latter more\n# convenient\nclass ModelHandler:\n    def __init__(self, model, scalers):\n        self.model  = model\n        self.scaler_t, self.scaler_x = scalers\n        \n        self.scale  = self.scaler_t.scale_[0] # for output\n        self.mean   = self.scaler_t.mean_[0]  # for output\n        self.fields = self.scaler_x.feature_names_in_\n        \n    def __call__(self, df):\n        \n        # scale input data\n        x  = np.array(self.scaler_x.transform(df[self.fields]))\n        x  = torch.Tensor(x)\n\n        # go to evaluation mode\n        self.model.eval()\n    \n        # compute,reshape to a 1d array, and convert to a numpy array\n        Y  = self.model(x).view(-1, ).detach().numpy()\n        \n        # rescale output\n        Y  = self.mean + self.scale * Y\n        \n        if len(Y) == 1:\n            return Y[0]\n        else:\n            return Y\n        \n    def show(self):\n        for name, param in self.model.named_parameters():\n            if param.requires_grad:\n                print(name, param.data)\n                print()\n        \ndef train(model, optimizer, avloss, getbatch,\n          train_x, train_t, \n          valid_x, valid_t,\n          batch_size, \n          n_iterations, traces, \n          step=50):\n    \n    # to keep track of average losses\n    xx, yy_t, yy_v = traces\n    \n    n = len(valid_x)\n    \n    print(\'Iteration vs average loss\')\n    print("%10s\\t%10s\\t%10s" % \\\n          (\'iteration\', \'train-set\', \'valid-set\'))\n    \n    for ii in range(n_iterations):\n\n        # set mode to training so that training specific \n        # operations such as dropout are enabled.\n        model.train()\n        \n        # get a random sample (a batch) of data (as numpy arrays)\n        batch_x, batch_t = getbatch(train_x, train_t, batch_size)\n        \n        # convert the numpy arrays batch_x and batch_t to tensor \n        # types. The PyTorch tensor type is the magic that permits \n        # automatic differentiation with respect to parameters. \n        # However, since we do not need to take the derivatives\n        # with respect to x and t, we disable this feature\n        with torch.no_grad(): # no need to compute gradients \n            # wrt. x and t\n            x = torch.from_numpy(batch_x).float()\n            t = torch.from_numpy(batch_t).float()      \n\n        # compute the output of the model for the batch of data x\n        # Note: outputs is \n        #   of shape (-1, 1), but the tensor targets, t, is\n        #   of shape (-1,)\n        # In order for the tensor operations with outputs and t\n        # to work correctly, it is necessary that they have the\n        # same shape. We can do this with the reshape method.\n        outputs = model(x).reshape(t.shape)\n   \n        # compute a noisy approximation to the average loss\n        empirical_risk = avloss(outputs, t, x)\n        \n        # use automatic differentiation to compute a \n        # noisy approximation of the local gradient\n        optimizer.zero_grad()       # clear previous gradients\n        empirical_risk.backward()   # compute gradients\n        \n        # finally, advance one step in the direction of steepest \n        # descent, using the noisy local gradient. \n        optimizer.step()            # move one step\n        \n        if ii % step == 0:\n            \n            acc_t = validate(model, avloss, train_x[:n], train_t[:n]) \n            acc_v = validate(model, avloss, valid_x[:n], valid_t[:n])\n\n            if len(xx) < 1:\n                xx.append(0)\n                print("%10d\\t%10.6f\\t%10.6f" % \\\n                      (xx[-1], acc_t, acc_v))\n            else:\n                xx.append(xx[-1] + step)\n                print("\\r%10d\\t%10.6f\\t%10.6f" % \\\n                      (xx[-1], acc_t, acc_v), end=\'\')\n                \n            yy_t.append(acc_t)\n            yy_v.append(acc_v)\n    print()      \n    return (xx, yy_t, yy_v)\n\ndef plot_average_loss(traces, ftsize=18):\n    \n    xx, yy_t, yy_v = traces\n    \n    # create an empty figure\n    fig = plt.figure(figsize=(5, 5))\n    fig.tight_layout()\n    \n    # add a subplot to it\n    nrows, ncols, index = 1,1,1\n    ax  = fig.add_subplot(nrows,ncols,index)\n\n    ax.set_title("Average loss")\n    \n    ax.plot(xx, yy_t, \'b\', lw=2, label=\'Training\')\n    ax.plot(xx, yy_v, \'r\', lw=2, label=\'Validation\')\n\n    ax.set_xlabel(\'Iterations\', fontsize=ftsize)\n    ax.set_ylabel(\'average loss\', fontsize=ftsize)\n    ax.set_xscale(\'log\')\n    ax.set_yscale(\'log\')\n    ax.grid(True, which="both", linestyle=\'-\')\n    ax.legend(loc=\'upper right\')\n\n    plt.show()')


# In[20]:


import iqnutil as ut
importlib.reload(ut);


# ### Define model $f(\mathbf{x}, \theta)$
# 
# For simple models, it is sufficient to use the __Sequential__ class.

# In[21]:


get_ipython().run_cell_magic('writefile', 'iqn_model.py', '\nimport torch\nimport torch.nn as nn\n\nmodel = nn.Sequential(nn.Linear( 8, 50),\n                      nn.ReLU(),\n                      \n                      nn.Linear(50, 50),\n                      nn.ReLU(),\n                      \n                      nn.Linear(50, 50),\n                      nn.ReLU(), \n \n                      nn.Linear(50, 50),\n                      nn.ReLU(), \n \n                      nn.Linear(50, 1)) ')


# ### Train!

# In[22]:


import iqn_model as iqn
importlib.reload(iqn)
model = iqn.model
print(model)

n_batch       = 50
n_iterations  = 200000

learning_rate = 2.e-4
optimizer     = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate) 

traces = ([], [], [])
traces_step = 10

traces = ut.train(model, optimizer, 
                  ut.average_quantile_loss,
                  ut.get_batch,
                  train_x, train_t, 
                  valid_x, valid_t,
                  n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step)

ut.plot_average_loss(traces)

# save model parameter dictionary
torch.save(model.state_dict(), 'iqn_model.dict')


# In[17]:


n_batch       = 50
n_iterations  = 200000

traces = ut.train(model, optimizer, 
                  ut.average_quantile_loss,
                  ut.get_batch,
                  train_x, train_t, 
                  valid_x, valid_t,
                  n_batch, 
                  n_iterations,
                  traces,
                  step=traces_step)

ut.plot_average_loss(traces)

# save model parameter dictionary
torch.save(model.state_dict(), 'iqn_model400k.dict')


# In[18]:


dnn = ut.ModelHandler(model, scalers)


# ### Plot results of trained model

# In[69]:


def plot_model(df, dnn,
               gfile='fig_model.png', 
               fgsize=(6, 6), 
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
    ax.set_xlabel(r'$p_{T}$ (GeV)', fontsize=ftsize)

    ax.hist(df.RecoDatapT, 
            bins=xbins, 
            range=(xmin, xmax), alpha=0.3, color='blue')
   
    y = dnn(df)
    
    ax.hist(y, 
            bins=xbins, 
            range=(xmin, xmax), 
            alpha=0.3, 
            color='red')
    ax.grid()

    plt.tight_layout()
    plt.savefig(gfile)
    plt.show()


# In[70]:


plot_model(test_data, dnn)


# In[ ]:




