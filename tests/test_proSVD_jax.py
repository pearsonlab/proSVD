#%%
import jax
from jax import jit, random
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

#%%
n = 100
t = 10000

k = 6
l1 = k
chunk_size = 1
w_shift = None
decay_alpha = 1
trueSVD = True
history = 0
track_prev = True
mode = 'pro'

key = random.PRNGKey(0)
key, subkey1, subkey2 = random.split(key, 3)

A_init = random.normal(subkey1, shape=(n, l1))
A = random.normal(subkey2, shape=(n, t-l1))

# params which will be updated: Q, B (and U, S for true SVD)
Q, B = jnp.linalg.qr(A_init)
Q = Q[:, :k]
B = B[:k, :l1]

if trueSVD:
    U, S, _ = jnp.linalg.svd(A_init, full_matrices=False)
U = U[:, :k]
S = S[:k]

@jit
def rq(A):
    Q, R = jnp.linalg.qr(jnp.flipud(A).T)
    R = jnp.flipud(R.T)
    Q = Q.T
    return R[:, ::-1], Q[::-1, :]


# A is data size n,chunk_size
def pro_scan(carry, A):
    Q, B = list(map(carry.get, ['Q', 'B']))
    C = Q.T @ A
    A_perp = A - Q @ C
    Q_perp, B_perp = jnp.linalg.qr(A_perp, mode='reduced')

    Q_hat = jnp.concatenate((Q, Q_perp), axis=1)
    B_prev = jnp.concatenate((B, C), axis=1)
    tmp = jnp.zeros((B_perp.shape[0], B.shape[1]))
    tmp = jnp.concatenate((tmp, B_perp), axis=1)
    B_hat = jnp.concatenate((B_prev, tmp), axis=0)

    U, diag, V = jnp.linalg.svd(B_hat, full_matrices=False)
    diag *= decay_alpha

    # Gu_1, Tu = rq(U[:, :self.k])  # Baker et al.
    Mu = U[:k, :k] # proSVD
    U_tilde, _, V_tilde_T = jnp.linalg.svd(Mu, full_matrices=False)
    Tu = U_tilde @ V_tilde_T
    Gu_1 = U[:, :k] @ Tu.T
    
    Gv_1, Tv = rq(V[:, :k])

    # updating proSVD basis
    Q = Q_hat @ Gu_1
    B = Tu @ jnp.diag(diag[:k]) @ Tv.T

    # updating true SVD basis
    U, S, _ = jnp.linalg.svd(B)
    U = Q @ U

    newcarry = {'Q': Q,
                'B': B}
    accum = {'Qs': Q, 
             'Bs': B,
             'Us': U,
             'Ss': S, 
             'proj_pro': Q.T @ A,
             'proj_svd': U.T @ A}

    return newcarry, accum

init_carry = {'Q': Q, 
              'B': B}
A = A.T[..., None]
finalcarry, accum = jax.lax.scan(pro_scan, init_carry, A)


# %%
