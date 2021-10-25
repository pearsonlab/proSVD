#%%
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

import mdp # for incremental sfa, ica, pca
from proSVD import proSVD

# %% data from http://ieeg-swez.ethz.ch/
# sampled at 512 hz, each file is 3 mins before seizure, seizure, 3 mins after
file_loc = '/hdd/pgupta/ieeg_seizure/ID1/'
matdict = sio.loadmat(file_loc+'Sz1.mat')
data1 = matdict['EEG'].T  # transpose to sensors x time

matdict2 = sio.loadmat(file_loc+'Sz2.mat')
data2 = matdict2['EEG'].T

sr = 512 # sample rate, hz
s1_start = 3 * 60 * sr # 3 mins
s1_end = data1.shape[1] - s1_start

s2_start = data1.shape[1] # just seizure + s1_start # whole first + 3 mins
s2_end = data2.shape[1] - s1_start # whole first 
data = np.append(data1, data2[:, s1_start:s2_end], axis=1) # combining two seizures
# data = np.fliplr(data)
# center data?
# data = data - data.mean(axis=1)[:, np.newaxis]

plt.plot(data.T)

#%% spectrum
u, s, v = np.linalg.svd(data, full_matrices=False)
var = s**2

fig, ax = plt.subplots()
ax.plot(np.cumsum(var) / np.sum(var))
# ax.set_xlim((0, 10))

# %% proSVD
%%time
k = 10
l1 = 100
l = 100
num_iters = 1000 # int( (data.shape[1] - l1) / l)

pro = proSVD(k, trueSVD=True, history=num_iters, newMu=True)

pro.initialize(data[:, :l1])
t = l1
times = []
for i in range(num_iters):
    start = time.time()
    pro.updateSVD(data[:, t:t+l])
    times.append(time.time() - start)
    t += l

print('proSVD:\t', np.mean(times)*1000, np.std(times)*1000)

# %% incSFA (emphasis on SLOW)
%%time
num_iters = 1000

incsfa = mdp.nodes.IncSFANode(input_dim=data.shape[0], output_dim=k)

Q_incsfa = np.zeros(pro.Qs.shape)
incsfa.train(data[:, :l1].T)
Q_incsfa[:, :, 0] = incsfa.sf

t = l1
times = []
for i in range(num_iters):
    start = time.time()
    incsfa.train(data[:, t:t+l].T)
    times.append(time.time() - start)
    Q_incsfa[:, :, i+1] = incsfa.sf
    t += l

print('incSFA:\t', np.mean(times)*1000, np.std(times)*1000)

# %% CCI-PCA
%%time
# num_iters = 1000

ccipca = mdp.nodes.CCIPCANode(input_dim=data.shape[0], output_dim=k)

Q_ccipca = np.zeros(pro.Qs.shape)
ccipca.train(data[:, :l1].T)
Q_ccipca[:, :, 0] = ccipca.v

t = l1
times = []
for i in range(num_iters):
    start = time.time()
    ccipca.train(data[:, t:t+l].T)
    times.append(time.time() - start)
    Q_ccipca[:, :, i+1] = ccipca.v
    t += l

print('CCIPCA:\t', np.mean(times)*1000, np.std(times)*1000)

#%%
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(5,12), sharex=True, sharey=True)

derivs_incsfa = np.linalg.norm(Q_incsfa[:, :, 1:] - Q_incsfa[:, :, :-1], axis=0).T
ax1.plot(derivs_incsfa[100:, :], alpha=0.2)

derivs_ccipca = np.linalg.norm(Q_ccipca[:, :, 1:] - Q_ccipca[:, :, :-1], axis=0).T
ax2.plot(derivs_ccipca[100:, :], alpha=0.2)


derivsQ = pro.get_derivs()
derivsU = pro.get_derivs(trueSVD=True)
ax3.plot(derivsQ[100:], alpha=0.2)
# plt.plot(derivsU, color='gray', alpha=0.2, ls='--')


# distsQ = pro.get_dist_to_final()
# distsU = pro.get_dist_to_final(trueSVD=True)
# ax2.plot(distsQ[:, 100:].T)
# dotted lines
# for ax in [ax1, ax2]:
#     for time in [s1_start, s1_end, s2_start, s2_end]:
#         ax.axvline( (time / l) - l1, ls='--', color='gray')
# ax1.set(ylim=(-.005, 0.1))



# %%


fig, ax = plt.subplots(2, 1, figsize=(5,8))
labels = ['incsfa', 'prosvd']
ls = ['--', '-']
c = ['C0', 'C1']
for i, deriv in enumerate([derivs_incsfa, derivsQ]):
    for j in range(2):
        ax[j].plot(deriv[100:, j], label=labels[i], alpha=.5, ls=ls[i], c=c[i])
        ax[j].legend()
# %%
