#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
from matplotlib import cm; cmap = cm.Dark2
from scipy.ndimage import gaussian_filter1d

from proSVD import proSVD

#%% switching between different spectral regimes
np.random.seed(100)
n = 6
embed_dim = 60 # "neurons"
t_latent = [300]  # time
regime_changes = 10

fig, ax = plt.subplots(len(t_latent), 3, figsize=(20,4*len(t_latent)),
            gridspec_kw={'hspace': 0.5, 'wspace': 0.3})
if len(t_latent) == 1:
    ax = ax[np.newaxis, :]  # for plotting 1 row with ease

for row, curr_t in enumerate(t_latent):
    # generate data - keep U and V the same throughout
    U_true, _ = np.linalg.qr(np.random.normal(size=(n,n)))
    V_true, _ = np.linalg.qr(np.random.normal(size=(curr_t,curr_t)))
    V_true = gaussian_filter1d(V_true[:n, :], sigma=2)

    # define spectral regimes
    Ss_true = np.array([[1, 1, 0, 0, 1, 0],
                        [0, 0, 1, 1, 0, 1]])

    num_regimes = Ss_true.shape[0]
    noise = 0 # np.random.normal(scale=.05, size=(n,curr_t))
    regimes = [[] for i in range(num_regimes)]
    for i in range(num_regimes):
        S_true = Ss_true[i, :]
        regimes[i] = (U_true @ np.diag(S_true) @ V_true) + noise

    # append particular regime to true data
    latent_true = np.copy(regimes[-1])

    # picking which regimes go where
    # to switch every regime in order
    regime_order = np.arange(regime_changes) % num_regimes
    # completely random
    # regime_order = np.random.randint(num_regimes, size=regime_changes)
    # to sample a regime without replacement, then do it again once all regimes are sampled once
    # regime_order = np.array([np.random.choice(num_regimes, num_regimes, replace=False) for i in range(regime_changes)]).flatten()
    # regime_order = regime_order[:regime_changes+1]

    for curr_regime in regime_order:
        latent_true = np.append(latent_true, regimes[curr_regime], axis=1)
    regime_order = np.insert(regime_order, 0, num_regimes-1) # prepending first regime (for plotting later)
        
    # embed whole thing
    embed_mat = np.random.uniform(size=(embed_dim, n))
    embed_mat, R = np.linalg.qr(embed_mat) 
    embed_xs = np.dot(embed_mat, latent_true)
    xs = embed_xs

    # PROTOTYPE FOR DOING ANALYSIS ON ANY DATASET
    k = n
    l1 = xs.shape[0]
    l = 1
    num_iters = np.ceil((xs.shape[1] - l1) / l).astype('int')
    update_times = np.arange(1, num_iters) * l # index of when updates happen

    # window for explained vars (TODO: also use this for keeping window of history/projs)
    window = int(curr_t / 1)

    A_init = xs[:, :l1]
    pro = proSVD(k, history=num_iters, trueSVD=True)
    pro.initialize(A_init)

    projs = [np.zeros((k, xs.shape[1]-l1)) for i in range(2)]  # subtract l1 - init proj
    frac_vars = [np.zeros(projs[i].shape) for i in range(2)]
    derivs = np.zeros((k, num_iters))

    for i, t in enumerate(update_times):
        # proSVD update
        dat = xs[:, t:t+l]
        pro.updateSVD(dat)
        derivs[:, i] = np.linalg.norm(pro.curr_diff, axis=0)

        # getting proj and variance explained
        for j, basis in enumerate([pro.U, pro.Q]):
            projs[j][:, t:t+l] = basis.T @ dat
            # var explained over window
            if window > 0 and (t - l1) > window: # process more data than window before getting var
                curr_proj_vars = projs[j][:, t-l1-window:t-l1].var(axis=1)[:, np.newaxis]
                total_vars = xs[:, t-window:t].var(axis=1)
                frac_vars[j][:, t:t+l] = curr_proj_vars / total_vars.sum()
            else:  # cumulative variance
                curr_proj_vars = projs[j][:, :t].var(axis=1)[:, np.newaxis]
                total_vars = xs[:, l1:].var(axis=1)
                frac_vars[j][:, t:t+l] = curr_proj_vars / total_vars.sum()

    Qts, Scoll, Qcoll = (pro.Us, pro.Ss, pro.Qs)


    # QcollU = np.zeros((xs.shape[0], k, num_iters))
    # ScollU = np.zeros((xs.shape[0], num_iters))
    # projsU = np.zeros((k, xs.shape[1]-l1))
    # frac_varsU = np.zeros(projsU.shape)
    # t = 0
    # for i in range(l1, l1+num_iters):
    #     U, S, V_T = np.linalg.svd(xs[:, :t+l1+l], full_matrices=False)
    #     t = t + l
    #     currU = U[:, :k]
    #     QcollU[:, :, i-l1] = currU
    #     ScollU[:, i-l1] = S

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

    ax[row,0].plot(latent_true.T, alpha=.2)
    ax[row,0].set(xlabel='bins', ylabel='observed activity', title='observed signal')
    axtwin = ax[row,0].twinx()
    axtwin.plot(np.repeat(regime_order, curr_t), 
                label='variance of first 3 vectors', color='k')
    axtwin.set(ylabel='variance explained')

    titles = ['variance explained by \n left singular vectors', 
              'variance explained by \n proSVD basis vectors']
    for j in range(2):
        currax = ax[row, j+1]
        # individual and total var
        currax.plot(frac_vars[j][:, :t].T, alpha=0.2, color='k')
        # currax.plot(frac_vars[j][:, :t].sum(axis=0), ls='solid', color='k')

        # sum of vars for particular dimensions
        dims_to_sum = int(n / num_regimes) # assuming first regime is in first n / num_regimes dims
        labels = ['sum first 3 dims', 'sum last 3 dims']
        start = 0
        for p in range(num_regimes): 
            currax.plot(frac_vars[j][start:start+dims_to_sum, :t].sum(axis=0), lw=1.5,
                        ls='solid', color=cmap(p), label=labels[p])
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
            regime_start = curr_t * p
            text_x_loc = (regime_start + (curr_t)) / l # in data coords for ease
            currax.text(text_x_loc, 1.02, '{}'.format(regime_order[p]+1), 
                        transform=trans, color=cmap(1-regime_order[p])) # 1- for annoying color thing

    # dotted line indicating regime change
    # signal
    
    for j, currax in enumerate(ax[row, :]):
        z=1 if j==0 else l
        for lineloc in [(curr_t*(i+1))/z for i in range(regime_changes+1)]:
            lineloc += l1
            currax.axvline(lineloc, ls='--', color='grey', alpha=.6)
        

ax[0, 1].legend(loc='upper right')
axtwin.legend()
plt.suptitle('streaming SVD does not clearly identify spectral regime changes',
             y=1.12)

# plt.savefig('stream_svd_problem.png', bbox_inches='tight')

# %%
