import numpy as np
import torch; import torch.nn as nn
import pandas as pd


data = np.load('smallData.npy')
data=data.T
data, data.shape


rawRecoData=data[0:4,:].T
pd.DataFrame(rawRecoData).head()


recoData = data[4:8,:].T
pd.DataFrame(rawRecoData).head()


genData = data[8:12,:].T
pd.DataFrame(genData).head()


df = pd.DataFrame(data[0:12,:].T)

levels = ['rawRecoData', 'RecoData', 'genData']
kinematics=['pT','eta','phi','m']

columns = [level+k for level in levels for k in kinematics]
columns


df.head()


df.columns = columns
df.head(), df.shape


tau = np.random.uniform(0,1, size = df.shape[0])
tau


df['tau'] = tau
df.head()


df.to_csv('Data.csv')


data = pd.read_csv('Data.csv')
recpt = data['RecoDatapT']
genpt= data['genDatapT']
recpt


rechist, bins = np.histogram(recpt, bins=100)
genhist, _ = np.histogram(genpt, bins=100)


np=  rechist/(genhist+1e-10)
plt.hist(np)
plt.ylabel(r'$p_T^{rec}/p_T^{gen}$', fontsize=14)



