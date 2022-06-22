#A.A.

import numpy as np
from matplotlib_inline import backend_inline
backend_inline.set_matplotlib_formats('svg')
import numpy as np
import pandas as pd
import matplotlib
import matplotlib as mp

import matplotlib.pyplot as plt


font = {'family' : 'serif',
        'size'   : 10}
matplotlib.rc('font', **font)
matplotlib.rcParams.update({
    "text.usetex": True})

# ###############Original plotting (works but not like Braden's)




# def plot_all():
#     fig,ax = plt.subplots(2,2,figsize=(6,6))
#     ax=ax.flatten()

    




#     ######pT

#     ax[0].hist(JETS_DICT['Predicted_RecoDatapT']['dist'], label=JETS_DICT['Predicted_RecoDatapT']['label'],
#                bins=bins, alpha=0.3,density=True,color="#d7301f",range=pt_range)
#     ax[0].hist(JETS_DICT['Real_RecoDatapT']['dist'], label=JETS_DICT['Real_RecoDatapT']['label'],bins=bins, alpha=0.3, density=True,color="k",range=pt_range)
#     ax[0].set_xlabel(JETS_DICT['Predicted_RecoDatapT']['xlabel'])
#     ########eta
#     ax[1].hist(JETS_DICT['Predicted_RecoDataeta']['dist'], label=JETS_DICT['Predicted_RecoDataeta']['label'],bins=bins, alpha=0.3,density=True,color="#d7301f")
#     ax[1].hist(JETS_DICT['Real_RecoDataeta']['dist'], label=JETS_DICT['Real_RecoDataeta']['label'],bins=bins, alpha=0.3, density=True,color="k")
#     ax[1].set_xlabel(JETS_DICT['Predicted_RecoDataeta']['xlabel'])    

#     #######phi
#     ax[2].hist(JETS_DICT['Predicted_RecoDataphi']['dist'], label=JETS_DICT['Predicted_RecoDataphi']['label'],bins=bins, alpha=0.3,density=True,color="#d7301f")
#     ax[2].hist(JETS_DICT['Real_RecoDataphi']['dist'], label=JETS_DICT['Real_RecoDataphi']['label'],bins=bins, alpha=0.3, density=True,color="k")
#     ax[2].set_xlabel(JETS_DICT['Predicted_RecoDataphi']['xlabel'])    

#     #############m
#     ax[3].hist(JETS_DICT['Predicted_RecoDatam']['dist'], label=JETS_DICT['Predicted_RecoDatam']['label'],bins=bins, alpha=0.3,density=True,color="#d7301f",range=m_range)
#     ax[3].hist(JETS_DICT['Real_RecoDatam']['dist'], label=JETS_DICT['Real_RecoDatam']['label'],bins=bins, alpha=0.3, density=True,color="k",range=m_range)
#     ax[3].set_xlabel(JETS_DICT['Predicted_RecoDatam']['xlabel'])    


#     for i in range(4):
#         ax[i].legend()

#     plt.tight_layout()
    
#     # plt.savefig('IQN_All_consecutive.svg')

# plot_all()

############Start here for Braden-like plots

data = pd.read_csv('Data.csv')
data = data[['RecoDatapT','RecoDataeta','RecoDataphi','RecoDatam']]
data.columns = ['realpT','realeta','realphi','realm']

print(data.shape)
print()
predicted_data_path='predicted_data/consecutive/'

print(pd.read_csv(predicted_data_path+'RecoDatapT_predicted_consecutive.csv')['RecoDatapT_predicted'].shape)

norm_data=data.shape[0]
norm_IQN=pd.read_csv(predicted_data_path+'RecoDatapT_predicted_consecutive.csv')['RecoDatapT_predicted'].shape[0]
print('norm_data',norm_data,'\nnorm IQN',norm_IQN)

############


bins=50

JETS_DICT  = {
                    'Predicted_RecoDatapT' : {
                            'dist':pd.read_csv(predicted_data_path+'RecoDatapT_predicted_consecutive.csv')['RecoDatapT_predicted'],
                           'xlabel': r'$p_T$ (GeV)', 
                            'range':[0,100],
                           'label':'IQN'},

                    'Real_RecoDatapT' : {
                            'dist':data['realpT'],
                           'xlabel': r'$p_T$ (GeV)', 
                            'range':[0,100],
                           'label':'Data'},
           ############
                'Predicted_RecoDataeta' : {
                    'dist':pd.read_csv(predicted_data_path+'RecoDataeta_predicted_consecutive.csv')['RecoDataeta_predicted'],
                    'xlabel': r'$\eta$', 
                    'range':[-5,5],
                    'label':'IQN'},

                    'Real_RecoDataeta' : {
                            'dist':data['realeta'],
                           'xlabel': r'$\eta$', 
                            'range':[-5,5],
                           'label':'Data'},
            ##########################    
                'Predicted_RecoDataphi' : {
                    'dist':pd.read_csv(predicted_data_path+'RecoDataphi_predicted_consecutive.csv')['RecoDataphi_predicted'],
                    'xlabel': r'$\phi$', 
                    'range':[-3.2,3.2],
                    'label':'IQN'},

                    'Real_RecoDataphi' : {
                            'dist':data['realphi'],
                           'xlabel': r'$\phi$', 
                            'range':[-3.2,3.2],
                           'label':'Data'},
            ###################################
                'Predicted_RecoDatam' : {
                    'dist':pd.read_csv(predicted_data_path+'RecoDatam_predicted_consecutive.csv')['RecoDatam_predicted'],
                    'xlabel': r'$m$ (GeV)', 
                    'range':[0,25],
                    'label':'IQN'},

                    'Real_RecoDatam' : {
                            'dist':data['realm'],
                           'xlabel': r'$m$ (GeV)', 
                            'range':[0,25],
                           'label':'Data'},

          }


labels=["$p_T$ (GeV)", "$\eta$", "$\phi$", "mass (GeV)"]
titles=["pT", "eta", "phi", "mass"]
ranges= [[19.9,80],[-5,5],[-3.2,3.2],[0,25]]
allCounts=[]
allEdges=[]

def get_hist(label):
  """label could be "pT", "eta", "phi", "m"
  """
  predicted_label_counts, label_edges = np.histogram(JETS_DICT['Predicted_RecoData'+label]['dist'], range=JETS_DICT['Predicted_RecoData'+label]['range'], bins=bins)
  real_label_counts, _ = np.histogram(JETS_DICT['Real_RecoData'+label]['dist'], range=JETS_DICT['Real_RecoData'+label]['range'], bins=bins)
  label_edges = label_edges[1:]/2+label_edges[:-1]/2

  return real_label_counts, predicted_label_counts, label_edges

real_label_counts_pT, predicted_label_counts_pT, label_edges_pT = get_hist('pT')
real_label_counts_eta, predicted_label_counts_eta, label_edges_eta = get_hist('eta')
real_label_counts_phi, predicted_label_counts_phi, label_edges_phi = get_hist('phi')
real_label_counts_m, predicted_label_counts_m, label_edges_m = get_hist('m')

# fig2 = plt.figure(constrained_layout=True)
# spec2 = gridspec.GridSpec(ncols=2, nrows=2, figure=fig2)
# f2_ax1 = fig2.add_subplot(spec2[0, 0])
# f2_ax2 = fig2.add_subplot(spec2[0, 1])
# f2_ax3 = fig2.add_subplot(spec2[1, 0])
# f2_ax4 = fig2.add_subplot(spec2[1, 1])

# figpt, (ax1pt, ax2pt) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
# plt.tight_layout()
# figpt.subplots_adjust(wspace=0.0, hspace=0.1)
# figpt.show()
# figeta, (ax1eta, ax2eta) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
# plt.tight_layout()
# figeta.subplots_adjust(wspace=0.0, hspace=0.1)
# figeta.show()

def plot_one_pT():
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
  ax1.step(label_edges_pT, real_label_counts_pT/norm_data, where="mid", color="k", linewidth=0.5)# step real_count_pt
  ax1.step(label_edges_pT, predicted_label_counts_pT/norm_IQN, where="mid", color="#D7301F", linewidth=0.5)# step predicted_count_pt
  ax1.scatter(label_edges_pT, real_label_counts_pT/norm_data, label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
  ax1.scatter(label_edges_pT,predicted_label_counts_pT/norm_IQN, label="predicted", color="#D7301F", marker="x", s=5, linewidth=0.5)
  ax1.set_xlim(ranges[0])
  ax1.set_ylim(0, max(predicted_label_counts_pT/norm_IQN)*1.1)
  ax1.set_ylabel("counts")
  ax1.set_xticklabels([])
  ax1.legend(loc='upper right')
  
  ratio=(predicted_label_counts_pT/norm_IQN)/(real_label_counts_pT/norm_data)
  ax2.scatter(label_edges_pT, ratio, color="r", marker="o",facecolors="none", s=5, linewidth=0.5)#PREDICTED (IQN)/Reco (Data)
  ax2.scatter(label_edges_pT, ratio/ratio, color="k", marker="o",facecolors="none", s=5, linewidth=0.5)
  ax2.set_xlim(ranges[0])
  ax2.set_xlabel(labels[0])
  ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{reco}}$")
  ax2.set_ylim((0.9,1.1))
  ax2.set_xlim(ranges[0])
  plt.tight_layout()
  fig.subplots_adjust(wspace=0.5, hspace=0.2)
  fig.subplots_adjust(wspace=0.0, hspace=0.1)
  plt.savefig('images/like_braden/pT_IQN_100k_LikeBrade.png')
  plt.show(); fig.show()
  
  plt.axis('off')
  plt.gca().set_position([0, 0, 1, 1])
  #test for svg later
  # plt.savefig('pT_IQN_100k_LikeBrade.svg',format="svg")
  
################################ ETA ################################
def plot_one_eta():
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
  ax1.step(label_edges_eta, real_label_counts_eta/norm_data, where="mid", color="k", linewidth=0.5)# step real_count_pt
  ax1.step(label_edges_eta, predicted_label_counts_eta/norm_IQN, where="mid", color="#D7301F", linewidth=0.5)# step predicted_count_pt
  ax1.scatter(label_edges_eta, real_label_counts_eta/norm_data, label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
  ax1.scatter(label_edges_eta,predicted_label_counts_eta/norm_IQN, label="predicted", color="#D7301F", marker="x", s=5, linewidth=0.5)
  ax1.set_xlim(ranges[1])
  ax1.set_ylim(0, max(predicted_label_counts_eta/norm_IQN)*1.1)
  ax1.set_ylabel("counts")
  ax1.set_xticklabels([])
  ax1.legend(loc='upper right')
  
  ratio=(predicted_label_counts_eta/norm_IQN)/(real_label_counts_eta/norm_data)
  ax2.scatter(label_edges_eta, ratio, color="r", marker="o",facecolors="none", s=5, linewidth=0.5)#PREDICTED (IQN)/Reco (Data)
  ax2.scatter(label_edges_eta, ratio/ratio, color="k", marker="o",facecolors="none", s=5, linewidth=0.5)
  ax2.set_xlim(ranges[1])
  ax2.set_xlabel(labels[1])
  ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{reco}}$")
  ax2.set_ylim((0.9,1.1))
  ax2.set_xlim(ranges[1])
  plt.tight_layout()
  fig.subplots_adjust(wspace=0.5, hspace=0.2)
  fig.subplots_adjust(wspace=0.0, hspace=0.1)
  plt.savefig('images/like_braden/eta_IQN_100k_LikeBrade.png')
  plt.show(); fig.show()
  
  plt.axis('off')
  plt.gca().set_position([0, 0, 1, 1])
  #test for svg later
  # plt.savefig('pT_IQN_100k_LikeBrade.svg',format="svg")



####################################PHI##################
def plot_one_phi():
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
  ax1.step(label_edges_phi, real_label_counts_phi/norm_data, where="mid", color="k", linewidth=0.5)# step real_count_pt
  ax1.step(label_edges_phi, predicted_label_counts_phi/norm_IQN, where="mid", color="#D7301F", linewidth=0.5)# step predicted_count_pt
  ax1.scatter(label_edges_phi, real_label_counts_phi/norm_data, label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
  ax1.scatter(label_edges_phi,predicted_label_counts_phi/norm_IQN, label="predicted", color="#D7301F", marker="x", s=5, linewidth=0.5)
  ax1.set_xlim(ranges[2])
  ax1.set_ylim(0, max(predicted_label_counts_phi/norm_IQN)*1.1)
  ax1.set_ylabel("counts")
  ax1.set_xticklabels([])
  ax1.legend(loc='upper right')
  
  ratio=(predicted_label_counts_phi/norm_IQN)/(real_label_counts_phi/norm_data)
  ax2.scatter(label_edges_phi, ratio, color="r", marker="o",facecolors="none", s=5, linewidth=0.5)#PREDICTED (IQN)/Reco (Data)
  ax2.scatter(label_edges_phi, ratio/ratio, color="k", marker="o",facecolors="none", s=5, linewidth=0.5)
  ax2.set_xlim(ranges[2])
  ax2.set_xlabel(labels[2])
  ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{reco}}$")
  ax2.set_ylim((0.9,1.1))
  ax2.set_xlim(ranges[2])
  plt.tight_layout()
  fig.subplots_adjust(wspace=0.5, hspace=0.2)
  fig.subplots_adjust(wspace=0.0, hspace=0.1)
  plt.savefig('images/like_braden/phi_IQN_100k_LikeBrade.png')
  plt.show(); fig.show()
  
  plt.axis('off')
  plt.gca().set_position([0, 0, 1, 1])
  
######################################MASS#####################################
def plot_one_m():
  fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(3.5*3/2.5,3.8), gridspec_kw={'height_ratios': [2,0.5]})
  ax1.step(label_edges_m, real_label_counts_m/norm_data, where="mid", color="k", linewidth=0.5)# step real_count_pt
  ax1.step(label_edges_m, predicted_label_counts_m/norm_IQN, where="mid", color="#D7301F", linewidth=0.5)# step predicted_count_pt
  ax1.scatter(label_edges_m, real_label_counts_m/norm_data, label="reco",  color="k",facecolors='none', marker="o", s=5, linewidth=0.5)
  ax1.scatter(label_edges_m,predicted_label_counts_m/norm_IQN, label="predicted", color="#D7301F", marker="x", s=5, linewidth=0.5)
  ax1.set_xlim(ranges[3])
  ax1.set_ylim(0, max(predicted_label_counts_m/norm_IQN)*1.1)
  ax1.set_ylabel("counts")
  ax1.set_xticklabels([])
  ax1.legend(loc='upper right')
  
  ratio=(predicted_label_counts_m/norm_IQN)/(real_label_counts_m/norm_data)
  ax2.scatter(label_edges_m, ratio, color="r", marker="o",facecolors="none", s=5, linewidth=0.5)#PREDICTED (IQN)/Reco (Data)
  ax2.scatter(label_edges_m, ratio/ratio, color="k", marker="o",facecolors="none", s=5, linewidth=0.5)
  ax2.set_xlim(ranges[3])
  ax2.set_xlabel(labels[3])
  ax2.set_ylabel(r"$\frac{\textnormal{predicted}}{\textnormal{reco}}$")
  ax2.set_ylim((0.9,1.1))
  ax2.set_xlim(ranges[3])
  plt.tight_layout()
  fig.subplots_adjust(wspace=0.5, hspace=0.2)
  fig.subplots_adjust(wspace=0.0, hspace=0.1)
  plt.savefig('images/like_braden/m_IQN_100k_LikeBrade.png')
  plt.show(); fig.show()
  
  plt.axis('off')
  plt.gca().set_position([0, 0, 1, 1])




plot_one_pT()
# plot_one_eta()
# plot_one_phi()
# plot_one_m()

