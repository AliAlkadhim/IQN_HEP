import numpy as np
import pandas as pd
import matplotlib as mp
import matplotlib.pyplot as plt

data = pd.read_csv('Data.csv')
data = data[['RecoDatapT','RecoDataeta','RecoDataphi','RecoDatam']]
data.columns = ['realpT','realeta','realphi','realm']

############
predicted_data_path='predicted_data/consecutive/'

JETS_DICT  = {
                    'Predicted_RecoDatapT' : {
                            'dist':pd.read_csv(predicted_data_path+'RecoDatapT_predicted_consecutive.csv')['RecoDatapT_predicted'],
                           'xlabel': r'$p_T$ (GeV)', 
                            'range':0,
                           'label':'IQN'},

                    'Real_RecoDatapT' : {
                            'dist':data['realpT'],
                           'xlabel': r'$p_T$ (GeV)', 
                            'range':0,
                           'label':'Data'},
           ############
                'Predicted_RecoDataeta' : {
                    'dist':pd.read_csv(predicted_data_path+'RecoDataeta_predicted_consecutive.csv')['RecoDataeta_predicted'],
                    'xlabel': r'$\eta$', 
                    'range':0,
                    'label':'IQN'},

                    'Real_RecoDataeta' : {
                            'dist':data['realeta'],
                           'xlabel': r'$\eta$', 
                            'range':0,
                           'label':'Data'},
            ##########################    
                'Predicted_RecoDataphi' : {
                    'dist':pd.read_csv(predicted_data_path+'RecoDataphi_predicted_consecutive.csv')['RecoDataphi_predicted'],
                    'xlabel': r'$\phi$', 
                    'range':0,
                    'label':'IQN'},

                    'Real_RecoDataphi' : {
                            'dist':data['realphi'],
                           'xlabel': r'$\phi$', 
                            'range':0,
                           'label':'Data'},
            ###################################
                'Predicted_RecoDatam' : {
                    'dist':pd.read_csv(predicted_data_path+'RecoDatam_predicted_consecutive.csv')['RecoDatam_predicted'],
                    'xlabel': r'$m$', 
                    'range':0,
                    'label':'IQN'},

                    'Real_RecoDatam' : {
                            'dist':data['realm'],
                           'xlabel': r'$m$', 
                            'range':0,
                           'label':'Data'},

          }


def plot_all():
    fig,ax = plt.subplots(2,2,figsize=(10,10))
    ax=ax.flatten()

    ######pT
    ax[0].hist(JETS_DICT['Predicted_RecoDatapT']['dist'], label=JETS_DICT['Predicted_RecoDatapT']['label'],bins=100, alpha=0.3,density=True)
    ax[0].hist(JETS_DICT['Real_RecoDatapT']['dist'], label=JETS_DICT['Real_RecoDatapT']['label'],bins=100, alpha=0.3, density=True)
    ax[0].set_xlabel(JETS_DICT['Predicted_RecoDatapT']['xlabel'])
    ########eta
    ax[1].hist(JETS_DICT['Predicted_RecoDataeta']['dist'], label=JETS_DICT['Predicted_RecoDataeta']['label'],bins=100, alpha=0.3,density=True)
    ax[1].hist(JETS_DICT['Real_RecoDataeta']['dist'], label=JETS_DICT['Real_RecoDataeta']['label'],bins=100, alpha=0.3, density=True)
    ax[1].set_xlabel(JETS_DICT['Predicted_RecoDataeta']['xlabel'])    

    #######phi
    ax[2].hist(JETS_DICT['Predicted_RecoDataphi']['dist'], label=JETS_DICT['Predicted_RecoDataphi']['label'],bins=100, alpha=0.3,density=True)
    ax[2].hist(JETS_DICT['Real_RecoDataphi']['dist'], label=JETS_DICT['Real_RecoDataphi']['label'],bins=100, alpha=0.3, density=True)
    ax[2].set_xlabel(JETS_DICT['Predicted_RecoDataphi']['xlabel'])    

    #############m
    ax[3].hist(JETS_DICT['Predicted_RecoDatam']['dist'], label=JETS_DICT['Predicted_RecoDatam']['label'],bins=100, alpha=0.3,density=True)
    ax[3].hist(JETS_DICT['Real_RecoDatam']['dist'], label=JETS_DICT['Real_RecoDatam']['label'],bins=100, alpha=0.3, density=True)
    ax[3].set_xlabel(JETS_DICT['Predicted_RecoDatam']['xlabel'])    


    for i in range(4):
        ax[i].legend()

    plt.tight_layout()
    plt.show()

plot_all()