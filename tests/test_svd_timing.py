#%%
import time
import numpy as np
import jax
from jax import jit, random
import jax.numpy as jnp
import matplotlib.pyplot as plt

jax.config.update('jax_platform_name', 'cpu')

#%%
jit_svd = jit(jnp.linalg.svd)

key = random.PRNGKey(0)

n_range = [50, 100, 250, 500, 1000]
n_repeats = 25
times = [[], []]
for i, curr_n in enumerate(n_range):
    a = random.normal(key, shape=(curr_n, curr_n))
    times_curr = [[], []]
    for j in range(n_repeats):
        npstart = time.time()
        np.linalg.svd(a)
        times_curr[0].append(time.time() - npstart)

        jnpstart = time.time()
        jit_svd(a)
        times_curr[1].append(time.time() - jnpstart)

        for k, meth in enumerate(['numpy', 'jax']):
            print(meth, '\t{:.3f}'.format(times_curr[k][-1]))
        print('------')
    times[0].append(times_curr[0])
    times[1].append(times_curr[1])


#%%
fig, ax = plt.subplots(1, len(n_range), figsize=(17,4))

means = [[], []]
stds = [[], []]
for i, curr_dim in enumerate(n_range):
    for j, label in enumerate(['numpy', 'jax']):
        currtimes = np.array(times[j][i])[1:]*1000
        ax[i].plot(currtimes, label=label)
        ax[i].set(title='{}-dim square SVD time'.format(curr_dim),
                  xlabel='iter', ylabel='time (ms)')
        means[j].append(currtimes.mean())
        stds[j].append(currtimes.std())
    ax[i].legend()

fig1, ax1 = plt.subplots()
for i, method in enumerate(['numpy', 'jax']):
    ax1.errorbar(x=n_range, y=means[i], yerr=stds[i], label=method)
ax1.legend()



# %%
