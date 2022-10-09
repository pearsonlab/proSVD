#%%
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from proSVD import proSVD

#%%
# make large data high-dimensional data
n = 100000 # number of samples
p = 1000 # dimension of sample
X = np.random.uniform(-1, 1, size=(n,p))
X -= X.mean(axis=0)[None, :]

# do PCA and SVD
pca = PCA()
startpca = time.time()
pca.fit(X)
print(f'sklearn PCA took {time.time()-startpca:.2f} s')

startsvd = time.time()
u, s, v = np.linalg.svd(X, full_matrices=False)
print(f'numpy svd took {time.time()-startsvd:.2f} s')

# eigenvalues of covariance = singular values squared / (n_samples-1)
print(np.allclose((s**2)/(n-1), pca.explained_variance_)) 

#%% 
# proSVD takes data in shape (num_dimensions, num_samples)
X = X.reshape((p, n))

# proSVD parameters
k = p # dimension to reduce to (keeping all p dims as example)
n_inits = 10000 # number of columns (samples) get initial basis
n_samples_update = n-n_inits # number of columns (samples) used per update iteration
decay_alpha = 1 # how much discounting of old data (sets effective window size, alpha=1 is all seen data)

# get number of iterations for entire dataset
num_iters = 1 # np.floor((X.shape[1]-n_inits-n_samples_update)/n_samples_update).astype('int')
update_times = np.arange(n_inits, num_iters*n_samples_update, n_samples_update) # index of when updates happen (not including init)

# make proSVD object, run
pro = proSVD(k, n_samples_update, decay_alpha=decay_alpha, trueSVD=True)
pro.initialize(X[:, :n_inits])
startpro = time.time()
for i, t in enumerate(update_times): 
    start, end = t, t+n_samples_update
    dat = X[:, start:end]

    pro.preupdate()
    pro.updateSVD(dat)
    pro.postupdate()
print(f'proSVD took {time.time()-startpro:.2f} s')

#%%
# visualize 
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(s, label='true SVD')
ax[0].plot(pro.S, label='proSVD')

ax[1].plot(pca.explained_variance_, label='PCA eigenvalues')
ax[1].plot((pro.S**2)/(n-1), label='proSVD eigenvalues')

for currax in ax:
    currax.legend()
    
#%%