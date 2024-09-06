#%% Run Packages

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import warnings
import sys
import numpy as np
import nibabel as nib
from nilearn.input_data import NiftiMasker,  MultiNiftiMasker
from scipy import stats
from scipy import signal
from sklearn import preprocessing
import matplotlib.pyplot as plt
import os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn
from functools import partial
import bayesian_changepoint_detection.generate_data as gd
from bayesian_changepoint_detection.priors import const_prior
from bayesian_changepoint_detection.offline_likelihoods import IndepentFeaturesLikelihood
from bayesian_changepoint_detection.bayesian_models import offline_changepoint_detection
from bayesian_changepoint_detection.offline_likelihoods import FullCovarianceLikelihood

#Set plotting settings
matplotlib.use('TkAgg')
plt.ioff()

#%% Import CSV Time Series File

temp_data = open(r'C:\Users\pseco\Documents\Dissertation Notes\Files\tpl-MNI152NLin2009cAsym_atlas-schaefer2011Combined_dseg.txt', 'r')
data_dict = {}
for line in temp_data.readlines():
    data_dict[line.split('\t')[0].strip()] = line.split('\t')[1].strip()

header_list = list(data_dict.values())

data_ts = data = pd.read_csv(r'C:\Users\pseco\Documents\Dissertation Notes\Files\sub-3615_task-rest_feature-corrMatrix_atlas-power2011_timeseries.tsv',
                                sep = '\t', names = header_list)

#%% Break Out Into Time Windows Data Sets


def static_time_windows(df, t, overlap):
    windows = []
    i = 0
    while i < df.shape[0]:

        # 1st window
        if i-overlap < 0:
            windows.append(df.iloc[i:i+t, :])

        # Last window
        elif i+t > df.shape[0]:
            windows.append(df.iloc[i:df.shape[0], :])

        # Other windows
        else:
            windows.append(df.iloc[i:i+t, :])

        i += t-overlap

    return windows


ts_windows = static_time_windows(data_ts, 25, 5)


#%%
'''
#Change Point Time Windows (works but not fast)


data_test = pd.read_csv(
    r'C:\Users\pseco\Documents\Dissertation Notes\Files\sub-3615_task-rest_feature-corrMatrix_atlas-power2011_timeseries.tsv',
    sep='\t', header=None)

def dynamic_time_windows(data):

    data_np = np.array(data)

    Q_full, P_full, Pcp_full = offline_changepoint_detection(
        data_np, partial(const_prior, p=1 / (len(data_np) + 1)), FullCovarianceLikelihood(), truncate=-20
    )

    # Plot data and possible change points
    fig, ax = plt.subplots(2, figsize=[18, 8])
    for d in range(data_np.shape[1]):
        ax[0].plot(data_np[:, d])
    plt.legend(['Raw data'])
    ax[1].plot(np.exp(Pcp_full).sum(0))
    plt.legend(['Full Covariance Model'])
    plt.show()

    return Pcp_full


change_points = dynamic_time_windows(data_test)


print('Stopped')

'''

#%% Compute Pairwise Squared Magnitude Coherence

test_data = np.array(data_ts)

coherence_array = []
for col in range(test_data.shape[1]):
    for col2 in range(test_data.shape[1]):
        freqs, coherence_measures = signal.coherence(x=test_data[:, col], y=test_data[:, col2],
                                                fs=1.0, nperseg=test_data.shape[0], noverlap=0)
        coherence_array.append(coherence_measures)

print('Finished')




#%%
