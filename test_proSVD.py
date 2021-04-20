import time
import numpy as np
import matplotlib.pyplot as plt

import proSVD
from utils import get_streamingSVD


def something():
    # %%time
    # experiment 6.3 - transients that dominate variance
    # xt = a1cos(ω1t)e1 + a2cos(ω2t)e2 + b(t)e4
    n_dims = 3
    embed_dims = 4
    a1, a2 = 5, 6
    om1, om2 = .15, .15
    num_steps = 500

    k = 3
    l1 = 3
    l = 1
    decay = 1
    num_iters = np.ceil((num_steps - l1) / l).astype('int')

    transients = [(100, 150), (250, 300), (400, 450)]
    bs = np.zeros((num_steps))
    for i in range(len(transients)):
        start, end = transients[i]
        bs[start:end] = np.random.uniform(-200, 200, size=(end - start)) # magnitude higher than a_j

    E, R = np.linalg.qr(np.random.uniform(size=(n_dims,4)))
    # E = np.identity(n_dims)
    e1, e2, e4 = E
    es = np.zeros((n_dims, 3, num_steps))
    for t in range(num_steps):
        es[:, :, t] = [a1*np.cos(om1*t)*e1, a2*np.cos(om2*t)*e2, bs[t]*e4]
    xs = np.sum(es, axis=1)

    # embedding in higher dimension
    # embed_mat = np.random.uniform(size=(embed_dim, n_dims))
    # E_embed = np.dot(embed_mat, E)
    # E_embed, R = np.linalg.qr(E_embed) 
    # embed_xs = np.dot(embed_mat, xs)
    # xs = embed_xs

    pro = proSVD.proSVD(k, history=xs.shape[1]-l1, trueSVD=True)
    pro.initialize(xs[:,:l1])
    for i in np.arange(l1,xs.shape[1]):
        pro.updateSVD(xs[:,i:i+1])
    Qtcoll, Scoll, Qcoll = (pro.Qts, pro.Ss, pro.Qs)

    QcollU = np.zeros((n_dims, k, num_iters))
    ScollU = np.zeros((n_dims, num_iters))
    t = 0
    for i in range(num_iters):
        U, S, V_T = np.linalg.svd(xs[:, :t+l1+l])
        t = t + l
        currU = U[:, :k]
        QcollU[:, :, i] = currU
        ScollU[:, i] = S
        
        
    # plotting
    widths = [1.5, 3, 3]
    gs_kw = dict(hspace=.1, wspace=.3, width_ratios=widths)
    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=(15, 4), sharey='col', sharex='col', gridspec_kw=gs_kw)
    gs = axs[0, 0].get_gridspec()
    # remove the underlying axes
    for i in range(3):
        for j in range(1,3):
            axs[i, j].remove()
            
    axs[0, 0].set_title('data with transient')
    axs[2, 0].set(xlabel='time')

    axbig1 = fig.add_subplot(gs[:, 1])
    axbig2 = fig.add_subplot(gs[:, 2])
        
    for i in range(xs.shape[0]):
        axs[i, 0].plot(xs[i, :])
        axs[i, 0].set(ylabel='sensor {}'.format(i+1))
        
    colors = ['blue', 'orange', 'red', 'green']
    for i in range(3):
        Qvecs = Qcoll[:, i, :]
        Qvecs_dists = Qvecs[:, 1:] - Qvecs[:, :-1]
        Qvecs_dists = np.linalg.norm(Qvecs_dists, axis=0)
        axbig2.plot(Qvecs_dists[Qvecs_dists < 1], color=colors[i], label='vector {}'.format(i+1))
        axbig2.set(title='ssSVD vectors adapt to transient',
                ylabel='derivative of ssSVD basis vectors', xlabel='timestep')
        
        Qvecs = QcollU[:, i, :]
        Qvecs_dists = Qvecs[:, 1:] - Qvecs[:, :-1]
        Qvecs_dists = np.linalg.norm(Qvecs_dists, axis=0)
        axbig1.plot(Qvecs_dists[Qvecs_dists < 1], color=colors[i], label='vector {}'.format(i+1))
        axbig1.set(title='streaming SVD vectors jump with each transient',
                ylabel='derivative of singular vectors', xlabel='timestep')
        
    axbig1.legend(loc='upper right')
    axbig2.legend(loc='upper right')


    # greying transients
    for start, end in transients:
        axs[2, 0].axvspan(start, end, alpha=.1, color='gray')
        axs[2, 0].axvline(start, alpha=.6, color='gray', ls='--')
        axs[2, 0].axvline(end, alpha=.6, color='gray', ls='--')

    for start, end in transients:
        start -= l1 + 1
        end -= l1 + 1
        for curr_ax in [axbig1, axbig2]:
            curr_ax.axvspan(start, end, alpha=.1, color='gray')
            curr_ax.axvline(start, alpha=.6, color='gray', ls='--')
            curr_ax.axvline(end, alpha=.6, color='gray', ls='--')
            curr_ax.set(ylim=(0, .05))
            
    # axbig1.set(ylim=(0, .05))
    # axbig2.set(ylim=(0, .05))        

    # plt.savefig('figures/cosyne2021/fig2.svg', bbox_inches='tight')
  
def test_proSVD():
    n, t = 20, 100
    A = np.random.uniform(size=(n, t))

    # streaming params
    k = 15
    l1 = k
    l = 20
    num_iters = int(np.ceil(t - l1) / l)

    # true svd
    Us, Ss = get_streamingSVD(A, k, l1, l, num_iters, window=False)

    # procrustean svd
    A_init = A[:, :l1]
    pro = proSVD.proSVD(k=15, history=num_iters, trueSVD=True)
    pro.initialize(A_init)
    for i in np.arange(l1, l1+num_iters, l):
        dat = A[:, i:i+1]
        pro.updateSVD(dat)
    Qts, Ss, Qs = (pro.Qts, pro.Ss, pro.Qs)

    W = pro.get_W(A_init)
    print(Qts.shape)

def get_timing_chunk_size(chunk_range, n=100): # given n channels, loops through chunk size
    k = 6
    l1 = k

    # procrustean svd
    A_init = np.random.uniform(size=(n, l1))
    pro = proSVD.proSVD(k=k, history=0, trueSVD=False) # lightweight
    pro.initialize(A_init)

    times = np.zeros(chunk_range.shape)
    for i, chunk_size in enumerate(chunk_range):
        dat = np.random.uniform(size=(n, chunk_size))
        # time the update
        start = time.time()
        pro._updateSVD(dat)
        end = time.time() - start
        times[i] = end * 1000

    return times

# timing for reducing channel_range  down to k=6 dims
# looping over chunk ranges
# iters to avg over
def test_timing(channel_range, chunk_range, iters=20):
    total_times = np.zeros((len(channel_range), len(chunk_range)))

    for i, n in enumerate(channel_range):
        avg_times = np.zeros((iters, len(chunk_range)))
        for j in range(iters):
            times = get_timing_chunk_size(chunk_range, n=n)
            avg_times[j, :] = times
        total_times[i, :] = avg_times.mean(axis=0)

    # if getting per-sample time
    channelchunks = np.outer(channel_range, chunk_range)
    total_times /= channelchunks

    fig, ax = plt.subplots(1, 1, figsize=(5, 4)) 
    fig.subplots_adjust(wspace=0.3)
    for i, n in enumerate(channel_range):
        times = total_times[i, :]
        ax.semilogy(chunk_range, times, label='n = {}'.format(n))
        ax.set(xlabel='chunk size', ylabel='total time (ms)')
        fig.suptitle('Time to reduce from $n$ dims to $k=6$ dims \n for increasing chunk size (number of columns)',
                y=1.02)
    ax.legend(title="Input data dimension")
    ax.legend()
    plt.show()
    # plt.savefig('timing.svg', bbox_inches='tight')


def test_W():
    n, t = 8, 30
    A = np.random.uniform(size=(n, t))

    # streaming params
    # TODO: figure out combinations that work and don't
    k = 2
    l1 = 2
    l = 5
    num_iters = int(np.ceil(t - l1) / l)

    # TODO: true svd for Vs as well
    # Us, Ss = get_streamingSVD(A, k, l1, l, num_iters, window=False)

    # proSVD
    A_init = A[:, :l1]
    pro = proSVD.proSVD(k=k, history=num_iters, trueSVD=True)
    pro.initialize(A_init)
    print(pro.Q.shape, pro.B.shape, pro.W.shape)

    t = l1
    for i in np.arange(num_iters):
        dat = A[:, t:t+l]
        pro.updateSVD(dat)
        print(pro.W.shape)
        t += l

    # recon = pro.Q @ pro.B @ pro.W.T
    # print(np.allclose(recon, A[:, :recon.shape[1]]))

def main():
    # test_proSVD()
    # channel_range = np.array([10, 50, 100, 200])
    # chunk_range = np.arange(1, 10, 1)
    # test_timing(channel_range, chunk_range)
    test_W()
    # something()

if __name__ == "__main__":
    main()

