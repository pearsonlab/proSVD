import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d
import scipy
import h5py

from matplotlib.cm import get_cmap
cmap = get_cmap('Dark2')
from matplotlib import animation 
from matplotlib.lines import Line2D
from IPython.display import HTML

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import proSVD

## behavior video (mouse face, paws)
file_loc = '../behave/reconstructed_Cam1_trials2_3.h5'    # from mSM49/SpatialDisc/30-Jul-2018/
with h5py.File(file_loc, 'r') as f:
    # breakpoint()
    data = np.array(f['cam1'])#[:,:,29800:30100] 
spikes = np.reshape(data, (data.shape[0]*data.shape[1], data.shape[2]))
## now spikes in shape assumed below, (pixels x some kind of time)

print('spikes shape ', spikes.shape)

## spiking data
# file_loc = '../behave/Neuropixels/N14/spikeTrace.mat'
# mat_str = 'spikeTrace' 

## using loadmat like this, syntax equals matlab struct (struct.field)
# mat_contents = sio.loadmat(file_loc, struct_as_record=False, squeeze_me=True)
# data = mat_contents[mat_str]
# spikes = data.T #neurons x time

# # smoothing spikes
spikes_smoothed = np.zeros(spikes.shape)
for i in range(spikes.shape[0]):
    spikes_smoothed[i, :] = gaussian_filter1d(spikes[i, :].astype('float'), sigma=2)
spikes = spikes_smoothed

# projecting smoothed spikes onto subspace discovered by svd on whole dataset
u, s, v = np.linalg.svd(spikes, full_matrices=False)
proj = u.T.dot(spikes)
print('did svd and projected')

############### doing streamingSVD on smoothed spikes, projecting spikes onto the learned subspace

l1 = 40    # num cols to init with
l = 1       # num cols processed per iter
decay = 1   # 'forgetting' to track nonstationarity. 1 = no forgetting
k = 100     # number of singular basis vectors to keep 
num_iters = np.ceil((spikes.shape[1] - l1) / l).astype('int') # num iters to go through once
print(num_iters)

pro = proSVD(k, history=spikes.shape[1]-l1, trueSVD=True)
pro.initialize(spikes[:,:l1])
for i in np.arange(l1,spikes.shape[1]):
    pro.updateSVD(spikes[:,i:i+1])
Qtcoll, Scoll, Qcoll = (pro.Qts, pro.Ss, pro.Qs)

print('did ssSVD ')

##### plotting

curr_spikes = spikes[:, 44:44+196] #1810:1850]
full_proj = Qcoll[:, :, -1].T.dot(curr_spikes)
print(full_proj.shape)
# breakpoint()
# timepts = [0,10000,20000,40000] #[0,125,475]
timepts = [0,50,100,195] 

fig, ax = plt.subplots(2, len(timepts), figsize=(15, 8), sharey='row', sharex='row')


for i, t in enumerate(timepts):
    trial_label = t + l1
    
    
    Q = Qcoll[:, :, t]
    S = Scoll[:, t]
    
    # U is U_inf after ssSVD on whole dataset. QU is rotated Q, and (QU)^T is projection
    # curr_proj = U.T.dot(Q.T).dot(curr_spikes)

    curr_proj = Q.T.dot(curr_spikes)

    ax[0, i].plot(curr_proj[0, :], curr_proj[1, :], color=cmap(0))
    ax[0, i].scatter(curr_proj[0, 0], curr_proj[1, 0], color=cmap(0))
    ax[0, i].set(title='{} bins ({} s) seen'.format(trial_label, (trial_label * 15 / 1000)), 
              xlabel='ssSVD basis vector 1', ylabel='ssSVD basis vector 2')

    ax[0, i].plot(full_proj[0, :], full_proj[1, :], c='k', alpha=.5, ls='--')
    ax[0, i].scatter(full_proj[0, 0], full_proj[1, 0], c='k')
    


full_proj = Qtcoll[:, :, -1].T.dot(curr_spikes)
for i, t in enumerate(timepts):
    trial_label = t + l1
    
    Q = Qtcoll[:, :, t]
    S = Scoll[:, t]
    
    # rotating Q to be more aligned with whole svd u
    # T = l2proc_min(Q, u)
    # Q = Q.T.dot(T)
     
    # U is U_inf after ssSVD on whole dataset. QU is rotated Q, and (QU)^T is projection
    # curr_proj = U.T.dot(Q.T).dot(curr_spikes)

    curr_proj = Q.T.dot(curr_spikes)

    ax[1, i].plot(curr_proj[0, :], curr_proj[1, :], color=cmap(1))
    ax[1, i].scatter(curr_proj[0, 0], curr_proj[1, 0], color=cmap(1))
    ax[1, i].set(xlabel='singular vector 1', ylabel='singular vector 2')

    ax[1, i].plot(full_proj[0, :], full_proj[1, :], c='k', alpha=.5, ls='--')
    ax[1, i].scatter(full_proj[0, 0], full_proj[1, 0], c='k')
    
    
custom_lines = [Line2D([0], [0], color=cmap(0)),
                Line2D([0], [0], color='k', ls='--')]
ax[0, 1].legend(custom_lines, ['ssSVD', 'whole data ssSVD'], loc='lower right')

custom_lines = [Line2D([0], [0], color=cmap(1)),
                Line2D([0], [0], color='k', ls='--')]
ax[1, 1].legend(custom_lines, ['streaming SVD', 'whole data SVD'], loc='lower left')

fig.suptitle('Projection of a single trial onto ssSVD basis vectors (top) or singular vectors (bottom)', y=.94)

plt.show()

breakpoint()