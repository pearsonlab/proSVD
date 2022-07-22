#%%
import time
import numpy as np
import matplotlib.pyplot as plt

import proSVD
from utils import get_streamingSVD

#%%
  
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
    return Us, Ss, pro

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
        ax.loglog(chunk_range, times, label='n = {}'.format(n), marker='o')
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

#%%
def test_recon():
    n, t = 3, 210
    A = np.random.normal(size=(n, t))

    k = 3
    l1 = 3
    l = 3
    num_iters = int(np.ceil(t - l1) / l)

    # TODO: true svd for Vs as well
    # Us, Ss = get_streamingSVD(A, k, l1, l, num_iters, window=False)

    # proSVD
    A_init = A[:, :l1]
    pro = proSVD.proSVD(k=k, history=num_iters, trueSVD=True)
    pro.initialize(A_init)
    # print(pro.Q.shape, pro.B.shape, pro.W.shape)

    t = l1
    for i in np.arange(num_iters):
        dat = A[:, t:t+l]
        pro.updateSVD(dat)
        # print(pro.W.shape)
        t += l

    recon = pro.Q @ pro.B @ pro.W.T
    print(A.shape)
    print(recon.shape)
    # print(np.allclose(A, recon))
    return pro, A

#%%
def main():
    # test_proSVD()
    channel_range = np.array([50, 100, 250, 500])
    chunk_range = np.array([1, 5, 25, 100, 250, 500])
    test_timing(channel_range, chunk_range)
    # test_W()
    # test_recon()

if __name__ == "__main__":
    main()

#%%