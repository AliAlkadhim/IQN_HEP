import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data = pd.read_csv('../Data.csv')
data = data[['RecoDatapT','RecoDataeta','RecoDataphi','RecoDatam']]
data.head()


data.columns = ['realpT','realera','realphi','realm']
data.head()


data.shape


predicted_pt=pd.read_csv('../predicted_data/RecoDatapT_predicted.csv')['RecoDatapT_predicted']
predicted_eta=pd.read_csv('../predicted_data/RecoDataeta_predicted.csv')['RecoDataeta_predicted']
predicted_phi=pd.read_csv('../predicted_data/RecoDataphi_predicted.csv')['RecoDataphi_predicted']
predicted_m=pd.read_csv('../predicted_data/RecoDatam_predicted.csv')['RecoDatam_predicted']
predicted=pd.concat([predicted_pt, predicted_eta, predicted_phi, predicted_m],axis=1)
predicted.columns=['predicted_pT','predicted_eta','predicted_phi','predicted_m']
predicted.head()


predicted.shape


# fig, ax = plt.subplots(1,4, figsize=(10,10))
# ax = ax.flatten()
range_pt=(20,60)
plt.hist(data['realpT'],bins=100,label='real pT',alpha=0.3,density=True,range=range_pt)
plt.hist(predicted['predicted_pT'],bins=100,label='predicted pT',alpha=0.3,density=True,range=range_pt)
plt.legend()


from scipy.special import rel_entr
KLD = rel_entr(data['realpT'][:20000],predicted['predicted_pT'])
print('KL= ',sum(KLD))


KLD = rel_entr(predicted['predicted_pT'],data['realpT'][:20000])
print('KL= ',sum(KLD))


plt.hist(data['realphi'],bins=100,label='real pT',alpha=0.3)
plt.hist(predicted['predicted_phi'],bins=100,label='predicted pT',alpha=0.3)
plt.legend()



