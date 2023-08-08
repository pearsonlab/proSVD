#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import cm; cmap = cm.Dark2
from scipy.ndimage import gaussian_filter1d

from proSVD import proSVD
from pro_utils import get_streamingSVD

#%% switching between different spectral regimes
np.random.seed(100)
n = 6
embed_dim = 100 # "neurons"
t_latent = 300  # time
regime_changes = 8

# define spectral regimes
Ss_true = np.array([[1, 1, 0, 0, 1, 0],
                    [0, 0, 1, 1, 0, 1]])

# generate data - keep U and V the same throughout
U_true, _ = np.linalg.qr(np.random.normal(size=(n,n)))
V_true, _ = np.linalg.qr(np.random.normal(size=(t_latent,t_latent)))
V_true = gaussian_filter1d(V_true[:n, :], sigma=2)
num_regimes = Ss_true.shape[0]
noise = 0 # np.random.normal(scale=.005, size=(n,t_latent))
regimes = [[] for i in range(num_regimes)]
for i in range(num_regimes):
    S_true = Ss_true[i, :]
    regimes[i] = (U_true @ np.diag(S_true) @ V_true) + noise

# picking which regimes go where
# to switch every regime in order
regime_order = np.arange(regime_changes) % num_regimes
# completely random
# regime_order = np.random.randint(num_regimes, size=regime_changes)
# to sample a regime without replacement, then do it again once all regimes are sampled once
# regime_order = np.array([np.random.choice(num_regimes, num_regimes, replace=False) for i in range(regime_changes)]).flatten()
# regime_order = regime_order[:regime_changes+1]
        
currmode = 'pro'
fig, ax = plt.subplots(1, 3, figsize=(20,4), gridspec_kw={'hspace': 0.5, 'wspace': 0.3})

# append particular regime to true data
latent_true = np.copy(regimes[-1])

for curr_regime in regime_order:
    latent_true = np.append(latent_true, regimes[curr_regime], axis=1)
regime_order = np.insert(regime_order, 0, num_regimes-1) # prepending first regime (for plotting later)
    
# embed whole thing
embed_mat = np.random.uniform(size=(embed_dim, n))
embed_mat, R = np.linalg.qr(embed_mat) 
embed_xs = np.dot(embed_mat, latent_true)
xs = np.array(embed_xs)

# PROTOTYPE FOR DOING ANALYSIS ON ANY DATASET
k = 6
l1 = xs.shape[0]
l = 1
decay = 1
num_iters = np.floor((xs.shape[1] - l1 - l)/l).astype('int')
print(num_iters)
update_times = np.arange(l1, num_iters*l, l) # index of when updates happen (not including init)

# window for explained vars (TODO: also use this for keeping window of history/projs)
window = int(t_latent / 1.5)

# create object
pro = proSVD(k, w_len=l, w_shift=None, decay_alpha=decay, history=num_iters, trueSVD=True, mode=currmode)
A_init = xs[:, :l1]
# init strategies:
# svd
# u, s, v = np.linalg.svd(A_init, full_matrices=False)
# Q_init = u[:, :k] # None
# B_init = np.diag(s[:k]) @ v[:k, :] # None
# random orthogonal decomposition
# Q_init = np.random.normal(size=(embed_dim, k))
# Q_init, B_init = np.linalg.qr(Q_init)
# B_init = Q_init.T @ A_init
# pro.initialize(A_init, Q_init=Q_init, B_init=B_init)

pro.initialize(A_init) # regular init
projs = [np.zeros((pro.k, xs.shape[1]-l1)) for i in range(2)]  # subtract l1 - init proj
frac_vars = [np.zeros(projs[i].shape) for i in range(2)]
derivs = np.zeros((pro.k, num_iters))
proj_mats = np.zeros((num_iters))

# UUs, Ss = get_streamingSVD(xs, k, l1, l, num_iters, window=False)

# run proSVD online
for i, t in enumerate(update_times): 
    start, end = t-l1, t+pro.w_len-l1
    dat = xs[:, start+l1:end+l1]

    # TODO: run should take user input pre/postupdate functions
    # they should be executed here
    pro.preupdate()
    pro.updateSVD(dat)
    pro.postupdate()

    # TODO: move this to postupdate
    # getting proj and variance explained
    for j, basis in enumerate([pro.U, pro.Q]):
        projs[j][:, start:end] = basis.T @ dat
        # var explained over window
        if window > 0 and end-window > 0: # process more data than window before getting var
            curr_proj_vars = projs[j][:, end-window:end].var(axis=1)[:, np.newaxis]
            total_vars = xs[:, end-window+l1:end+l1].var(axis=1)
            frac_vars[j][:, start:end] = curr_proj_vars / total_vars.sum()
        # else:  # cumulative variance
        #     curr_proj_vars = projs[j][:, :end].var(axis=1)[:, np.newaxis]
        #     total_vars = xs[:, l1:end].var(axis=1)
        #     frac_vars[j][:, start:end] = curr_proj_vars / total_vars.sum()
    # proSVD basis derivatives
    derivs[:, i] = np.linalg.norm(pro.Q-pro.Q_prev, axis=0)
    # proj mat changes
    basis1 = pro.Us[:, :, i]
    basis2 = pro.Us[:, :, i+1]
    proj_mats[i] = np.linalg.norm((basis1 @ basis1.T) - (basis2 @ basis2.T))

Qts, Scoll, Qcoll = (pro.Us, pro.Ss, pro.Qs)


# Us = np.zeros((xs.shape[0], k, num_iters))
# Ss = np.zeros((xs.shape[0], num_iters))
# projsU = np.zeros((k, xs.shape[1]-l1))
# frac_varsU = np.zeros(projsU.shape)
# t = 0
# for i in range(l1, l1+num_iters):
#     U, S, V_T = np.linalg.svd(xs[:, :t+l1+l], full_matrices=False)
#     t = t + l
#     currU = U[:, :k]
#     Us[:, :, i-l1] = currU
#     Ss[:, i-l1] = S

#     projsU[:, i-l1:i+1-l1] = basis.T @ dat
#     curr_proj_vars = projsU[:, :i].var(axis=1)[:, np.newaxis]
#     total_vars = xs[:, :i].var(axis=1)
#     frac_varsU[:, i:i+1] = curr_proj_vars / total_vars.sum()

t = derivs.shape[1]

# vals = (ScollU[:6, :]**2) / (ScollU[:, :]**2).sum(axis=0)
# ls = ['--', ':']
# cs = ['red', 'purple']
# for i in range(2):
#     ax[0].plot(frac_vars[i].sum(axis=0)[:t], ls=ls[i], color=cs[i])
# ax[0].plot(vals.sum(axis=0), color='k')
# ax[0].set(xlabel='bins seen', ylabel='total variance explained', 
#           title='problem with proSVD \n (due to precision?)')
# ax[0].legend(labels=['Q - pro', 'U - pro', 'U - dumb streaming'])
# ISSUE
# ax[0].plot(frac_varsU[:, :t].T)
# ax[0].set(title=titles[j], 
#             xlabel='bins seen', 
#             ylabel='fraction of variance explained',
#             ylim=(-.025, .55))

ax[0].plot(latent_true.T, alpha=.2)
ax[0].set(xlabel='bins', ylabel='observed activity', title='observed signal')
axtwin = ax[0].twinx()
axtwin.plot(np.repeat(regime_order, t_latent), 
            label='variance of first 3 vectors', color='k')
axtwin.set(ylabel='variance explained')

titles = ['variance explained by \n left singular vectors', 
        'variance explained by \n proSVD basis vectors']
for j in range(2):
    currax = ax[j+1]
    # individual and total var
    currax.plot(frac_vars[j][:, :t].T, alpha=0.2, color='k')
    # currax.plot(frac_vars[j][:, :t].sum(axis=0), ls='solid', color='k')

    # sum of vars for particular dimensions
    dims_to_sum = int(k / num_regimes) # assuming first regime is in first n / num_regimes dims
    labels = ['sum dims 1,2,3', 'sum dims 4,5,6']
    start = 0
    for p in range(num_regimes): 
        currax.plot(frac_vars[j][start:start+dims_to_sum, :t].sum(axis=0), lw=1.5,
                    ls='solid', color=cmap(p), label=labels[p])
        currax.plot(frac_vars[j][:, :t].sum(axis=0), color='k', ls='--')
        currax.set(ylim=(-.08, 1.1))
        start += dims_to_sum
    
    # labels
    currax.set_title(titles[j], y=1.08)
    currax.set(xlabel='bins seen', #'chunks seen (chunk size={})'.format(l), 
            ylabel='fraction of windowed variance explained')
                # ylim=(-.025, .55))

    # numbers on top indicating current regime
    for p in range(regime_changes+1): 
        trans = transforms.blended_transform_factory(currax.transData, 
                                                    currax.transAxes)
        regime_start = t_latent * p
        text_x_loc = (regime_start + (t_latent)-125) / l # in data coords for ease
        currax.text(text_x_loc, 1.02, '{}'.format(regime_order[p]+1), 
                    transform=trans, color=cmap(1-regime_order[p])) # 1- for annoying color thing

# dotted line indicating regime change
# signal
for j, currax in enumerate(ax[:]):
    for lineloc in [(t_latent*(i+1)-l1)/l for i in range(regime_changes+1)]:
        if j == 0:
            lineloc *= l
        currax.axvline(lineloc, ls='--', color='grey', alpha=.6)
        
ax[1].legend(loc='upper right')
axtwin.legend()
plt.suptitle('streaming SVD does not clearly identify spectral regime changes',
            y=1.12)
# plt.savefig('stream_svd_problem.png', bbox_inches='tight')

# u, s, v = np.linalg.svd(xs, full_matrices=False)
# u = u[:, :k]
# basis = pro.U
# res1 = np.linalg.norm(basis - u @ u.T @ basis)
# res2 = np.linalg.norm(u - basis @ basis.T @ u)
# print(res1, res2)



# plt.figure()
# plt.imshow(np.abs(u - pro.U))
# plt.colorbar()

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(pro.Ss.T)
labels = ['streaming SVD power shifts', 'proSVD power shifts']
for i in range(2):
    normdiffs = np.linalg.norm(np.diff(projs[i]**2, axis=1), axis=0)
    ax[i+1].plot(normdiffs, label=labels[i])
# ax[2].legend()

titles = ['singular values over time', 
        'changes in streaming SVD power:\n' + r'$ norm(diff((U^\top A)^2))$',
        'changes in proSVD power:\n' + r'$ norm(diff((Q^\top A)^2))$']
for i, currax in enumerate(ax):
    for p in range(regime_changes+1): 
        trans = transforms.blended_transform_factory(currax.transData, 
                                                        currax.transAxes)
        regime_start = t_latent * p
        text_x_loc = (regime_start + (t_latent)-125) / l # in data coords for ease
        currax.text(text_x_loc, 1.02, '{}'.format(regime_order[p]+1), 
                    transform=trans, color=cmap(1-regime_order[p])) # 1- for annoying color thing

    # dotted line indicating regime change
    for lineloc in [(t_latent*(i+1)-l1)/l for i in range(regime_changes+1)]:
        if j == 0:
            lineloc *= l
        currax.axvline(lineloc, ls='--', color='grey', alpha=.6)
    currax.set_title(titles[i], y=1.08)


titles = ['streaming SVD squared projections', 'proSVD squared projections']
for b in range(2):
    fig, ax = plt.subplots(3, 2, figsize=(10, 1.5*k))
    plt.subplots_adjust(hspace=.3)
    c = 0
    for w in range(2):
        for i, currax in enumerate(ax[:, w]):
            currax.plot((projs[b][c, :]**2).T, c='C{}'.format(c))
            for p in range(regime_changes+1): 
                trans = transforms.blended_transform_factory(currax.transData, 
                                                                currax.transAxes)
                regime_start = t_latent * p
                text_x_loc = (regime_start + (t_latent)-125) / l # in data coords for ease
                currax.text(text_x_loc, 1.02, '{}'.format(regime_order[p]+1), 
                            transform=trans, color=cmap(1-regime_order[p])) # 1- for annoying color thing

            # dotted line indicating regime change
            for lineloc in [(t_latent*(i+1)-l1)/l for i in range(regime_changes+1)]:
                if j == 0:
                    lineloc *= l
                currax.axvline(lineloc, ls='--', color='grey', alpha=.6)
            
            c += 1
    plt.suptitle(titles[b], y=.95)
# plt.savefig('stream_svd_problem2.png', bbox_inches='tight')

# %%

# %%
