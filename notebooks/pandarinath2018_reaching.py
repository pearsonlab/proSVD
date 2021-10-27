#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from scipy.ndimage import gaussian_filter1d

from proSVD.proSVD import proSVD
from proSVD.utils import get_streamingSVD, get_derivs

import os

# %%
# getting spikes aligned, concatenating all trials
def get_spikes(file_locs, bin_size=15):
    spikes_all = [[] for i in range(len(file_locs))]
    for sess, file_loc in enumerate(file_locs):
        # load data - stitch a few sessions together
        # using loadmat like this, syntax equals matlab struct (struct.field)
        mat_contents = sio.loadmat(file_loc, struct_as_record=False, squeeze_me=True)  
        data = mat_contents['data']

        align_to = 'GoCue' # 'Move' # 'GoCue' #
        num_steps = 1000
        num_trials = data.nTrials

        reduced_bins = int(num_steps / bin_size)
        trial_types = []
            
        reach_labels = dict(DownLeft=0, Left=1, UpLeft=2, Up=3, UpRight=4, Right=5, DownRight=6)
        spikes = np.zeros((data.nUnits, num_trials * reduced_bins))
        t = 0
        for i in range(num_trials):
            # all trials
            trial_types.append(data.targetDirectionName[i])
            
            if align_to == 'TargetOnset':
                start = data.TargetOnset[i]
            elif align_to == 'GoCue':
                start = data.GoCue[i]
            elif align_to == 'Move':
                start = data.Move[i] - 300
                
            end = start + num_steps
            curr_spikes = np.copy(data.spikeRasters[i, start:end, :].T)
            
            # reducing bin size
            spikes_reduced = np.zeros((data.nUnits, reduced_bins))
            for j in range(24):
                t1 = 0
                for k in range(spikes_reduced.shape[1]):
                    spikes_reduced[j, k] = np.sum(curr_spikes[j, t1:t1+bin_size])
                    t1 += bin_size
            spikes[:, t:t+reduced_bins] = spikes_reduced
            t += reduced_bins
            
        # smoothing spikes
        spikes_smoothed = np.zeros(spikes.shape)
        for i in range(spikes.shape[0]):
            spikes_smoothed[i, :] = gaussian_filter1d(spikes[i, :].astype('float'), sigma=2)
        spikes = spikes_smoothed

        # center data
        spikes -= spikes.mean(axis=1)[:, np.newaxis]
        spikes_all[sess] = spikes
    return spikes_all


file_dir = '/hdd/pgupta/lfads-neural-stitching-reproduce/export_v05_broadbandRethreshNonSorted_filtered/'
files = os.listdir(file_dir) # all sessions
files = [files[0]] # 1 session
# files = files[:2] # more sessions
file_locs = [file_dir + files[i] for i in range(len(files))]

bin_size = 15 # in ms
all_sess = get_spikes(file_locs, bin_size=bin_size)
data = all_sess[0]
for currsess in all_sess:
    data = np.append(data, currsess, axis=1)

#%% getting ordered spikes
# spikes_directions = [[] for i in range(len(reach_labels))]

# t = 0
# for i in range(num_trials):
#     if align_to == 'TargetOnset':
#         start = data.TargetOnset[i]
#     elif align_to == 'GoCue':
#         start = data.GoCue[i]
#     elif align_to == 'Move':
#         start = data.Move[i] - 300
        
#     end = start + num_steps
#     curr_spikes = np.copy(data.spikeRasters[i, start:end, :].T)
    
#     # reducing bin size
#     spikes_reduced = np.zeros((data.nUnits, reduced_bins))
#     for j in range(24):
#         t1 = 0
#         for k in range(spikes_reduced.shape[1]):
#             spikes_reduced[j, k] = np.sum(curr_spikes[j, t1:t1+bin_size])
#             t1 += bin_size
#     spikes_reduced
#     t += reduced_bins

#     currtype = reach_labels[data.targetDirectionName[i]]
#     spikes_directions[currtype].append(spikes_reduced.T)

# n_trials_directions = np.zeros((len(reach_labels)))
# for i in range(len(reach_labels)):
#     curr_n_trials = len(spikes_directions[i])
#     n_trials_directions[i] = curr_n_trials
#     spikes_directions[i] = np.array(spikes_directions[i]).reshape((curr_n_trials * reduced_bins, data.nUnits)).T

# spikes = np.zeros((data.nUnits, num_trials * reduced_bins))
# t = 0
# for i in range(len(reach_labels)):
#     bins_direction = int(n_trials_directions[i] * reduced_bins)
#     spikes[:, t:t+bins_direction] = spikes_directions[i]
#     t += bins_direction

#%% doing streamingSVD on smoothed spikes, projecting spikes onto the learned subspace

# PROTOTYPE FOR DOING ANALYSIS ON ANY DATASET
# params
k = 6  # reduced dim
l = 6    # num cols processed per iter
decay = 1 # 'forgetting' to track nonstationarity. 1 = no forgetting
l1 = k   # num cols init
num_iters = np.floor((data.shape[1] - l1) / l).astype('int') # num iters to go through data once
update_times = np.arange(1, num_iters) * l # index of when updates happen

# init
A_init = data[:, :l1]
pro = proSVD(k, history=num_iters, trueSVD=True)
pro.initialize(A_init)

# for svd and prosvd projections, variance explained
projs = [np.zeros((k, data.shape[1]-l1)) for i in range(2)]  # subtract l1 - init proj
frac_vars = [np.zeros(projs[i].shape) for i in range(2)]
# derivatives
derivs = np.zeros((k, num_iters))

# run proSVD online
for i, t in enumerate(update_times): 
    dat = data[:, t:t+l]
    pro.updateSVD(dat)
    # getting proj and variance explained
    for j, basis in enumerate([pro.U, pro.Q]):
        projs[j][:, t:t+l] = basis.T @ dat
        curr_proj_vars = projs[j][:, :t-l1].var(axis=1)[:, np.newaxis]
        total_vars = data[:, :t].var(axis=1)
        frac_vars[j][:, t:t+l] = curr_proj_vars / total_vars.sum()
    # proSVD basis derivatives
    derivs[:, i] = np.linalg.norm(pro.curr_diff, axis=0)


#%%
# derivs = get_derivs(pro)
t = derivs.shape[1]

fig, ax = plt.subplots(1, 3, figsize=(12,4))
ax[0].plot(derivs[:, :t].T)
ax[0].legend(labels=['vec{}'.format(i) for i in range(k)])
ax[0].set(title='derivatives of proSVD basis vectors', 
          xlabel='bins seen', ylabel='1st difference of basis vectors')

for j in range(2):
    ax[j+1].plot(frac_vars[j][:, :t].T)
    ax[j+1].plot(frac_vars[j].sum(axis=0)[:t], ls='--', color='k', alpha=0.5)
    ax[j+1].set(title='variance explained by each vector', 
            xlabel='bins seen', ylabel='fraction of variance explained')

ax[2].plot(frac_vars[0].sum(axis=0)[:t], ls=':', color='k', alpha=0.5)

#%% looking at singular values evolving
# fig, ax = plt.subplots()
# labels = ['sv{}'.format(i+1) for i in range(pro.Ss.shape[0])]
# ax.plot(pro.Ss.T)
# direction_change = np.cumsum(n_trials_directions * reduced_bins)
# for i in direction_change:
#     ax.axvline(i, ls='--', color='k', alpha=0.4)
# ax.set(xlabel='bin (15 ms bins)', ylabel='singular value')
# ax.legend(labels)

# # also change in svs
# Ss_deriv = pro.Ss[:, 1:] - pro.Ss[:, :-1]
# fig, axs = plt.subplots(2, 3, figsize=(12,7), sharex=True, sharey=True)
# t = 0
# for i in range(2):
#     for j in range(3):
#         axs[i, j].plot(Ss_deriv[i, :], color='C{}'.format(t))
#         for k in direction_change:
#             axs[i,j].axvline(k, ls='--', color='k', alpha=0.4)
#         t += 1
#         axs[i,j].set(ylim=(-.05, 1), title='sv{}'.format(t),
#                      xlabel='bin (15 ms)', ylabel='change in singular value')

# trial markers
# for currax in [ax1, ax2]:
    # currax.set(xlim=(-50, 5000))
    # for i in np.arange(0, spikes.shape[1], reduced_bins):
    #     currax.axvline(i, ls='--', color='grey', alpha=.4)

#%% doing full SVD
## %%time
Us, Ss = get_streamingSVD(data, data.shape[0], l1, l, num_iters, window=False)

#%% looking at stuff
vals = (Ss[:6, :]**2) / (Ss[:, :]**2).sum(axis=0)
for i in range(2):
    plt.plot(frac_vars[i].sum(axis=0)[:t])
plt.plot(vals.sum(axis=0), ls='--', color='k')
plt.set(xlabel='bins seen', ylabel='fraction of variance explained')
plt.legend(labels=['Q', 'U - pro', 'U - dumb streaming'])

# %% projecting each timepoint of neural activity onto the subspace learned for that chunk
all_projs_stream = np.zeros((pro.Qs.shape[1], pro.Qs.shape[2]*l))
for i in range(num_iters):
    Q = pro.Qs[:, :, i] # has first k components
    if i == 0:
        curr_neural = data[:, :l1]
        all_projs_stream[:, :l1] = Q.T @ curr_neural 
        t = l1
    else: 
        if t + l > Us.shape[2] * l:
            break
        # aligning neural to Q (depends on l1 and l)
        curr_neural = data[:, l1+((i-1)*l):l1+(i*l)]
        
        # projecting curr_neural onto curr_Q (our tracked subspace) and on full svd u
        all_projs_stream[:, t:t+l] = Q.T @ curr_neural
        t += l

all_projs_stream_true = np.zeros((Us.shape[1], Us.shape[2]*l))
for i in range(Us.shape[2]):
    Q = Us[:, :, i]
    if i == 0:
        curr_neural = data[:, :l1]
        all_projs_stream_true[:, :l1] = Q.T @ curr_neural
        t = l1
    else: 
        if t + l > Us.shape[2] * l:
            break
        # aligning neural to Q (depends on l1 and l)
        curr_neural = data[:, l1+((i-1)*l):l1+(i*l)]
        
        # projecting curr_neural onto curr_Q (our tracked subspace) and on full svd u
        all_projs_stream_true[:, t:t+l] = Q.T @ curr_neural
        t += l

num_remove = all_projs_stream.shape[1] - t
num_remove = all_projs_stream_true.shape[1] - t

if num_remove > 0:
    all_projs_stream_true = all_projs_stream_true[:, :-num_remove]
    all_projs_stream = all_projs_stream[:, :-num_remove]

# np.savez('neurips/ssSVD_results.npz', l1=l1, k=k, spikes=spikes, bin_size=bin_size, Us=Us, Qs=Qs)

# %%

np.savez('/hdd/pgupta/proSVD_results_reaching_long.npz', l1=l1, k=k, 
         spikes=data, bin_size=15, Us=Us, Qs=pro.Qs)

# %%
