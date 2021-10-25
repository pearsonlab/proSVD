import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# get derivs of Qs over iters (using history)
def get_derivs(pro, trueSVD=False):
    if trueSVD:
        derivs = np.linalg.norm(pro.Us[:, :, 1:] - pro.Us[:, :, :-1], axis=0).T
    else:
        derivs = np.linalg.norm(pro.Qs[:, :, 1:] - pro.Qs[:, :, :-1], axis=0).T
    return derivs

def get_dist_to_final(pro, trueSVD=False):
    if trueSVD:
        dists = np.linalg.norm(pro.Us - pro.Qt[:, :, np.newaxis], axis=0)
    else: 
        dists = np.linalg.norm(pro.Qs - pro.Q[:, :, np.newaxis], axis=0)
    return dists


def get_variances(pro, X_new):
    
    return #X_proj, X_proj.var(axis=1) 

# GENERATION
# random approach to generating stable dynamics matrix 
# for x'(t) = x(t)M
def get_stable_dynamics_mat(n, low=-.1, upp=.1):
    unstable = True
    while unstable:
        A = np.random.uniform(low, upp, size=(n,n))
#         A, _ = np.linalg.qr(A)
        evals, _ = np.linalg.eig(A)
        if np.max(evals) < 0:
            unstable = False
    return A

# generate stable, simple LDS
def generate_stable_LDS(n, t, M, noise_loc=0, noise_sig=1):
    X = np.zeros((n, t))
    X[:, 0] = np.random.uniform(-.001, .001, n) # setting x(0) randomly
    # M = np.random.uniform(-.1, .1, size=(n, n)) # random M - 
    # M, _ = np.linalg.qr(M)
    for i in range(1, t):
        noise = np.random.normal(noise_loc, noise_sig, size=(n,))
        X[:, i] = X[:, i-1] + ( X[:, i-1] @ M ) + noise
    return X

# embedding into larger N dim using orthogonal Q
def embed_data(X, N):
    n, t = X.shape
    A = np.random.uniform(size=(N,n)) # do this differently?
    Q, R = np.linalg.qr(A)
    return Q @ X

# ANALYSIS
# dumb streaming svd or window svd
def get_streamingSVD(X, k, l1, l, num_iters, window=True):
    Us = np.zeros((X.shape[0], k, num_iters))
    Ss = np.zeros((k, num_iters))
    
    U, S, V_T = np.linalg.svd(X[:, :l1], full_matrices=False)
    Us[:, :, 0] = U[:, :k]
    
    t = l1
    if window:
        start = t
        assert l >= k, "for windowed svd, chunk size should be >= k"
    else:
        start = 0
        
    for j in range(1, num_iters):
        U, S, V_T = np.linalg.svd(X[:, start:t+l], full_matrices=False)
        t = t + l
        Us[:, :, j] = U[:, :k]
        Ss[:, j] = S[:k]
        if window:
            start = t
    return Us, Ss

# PLOTTING
# plotting 3d plane given two vecs on the plane
def plot_3dplane(u, v, ax_lim, alpha=0.3, ax=None):
    xx, yy = np.meshgrid(np.linspace(-ax_lim, ax_lim, 10), 
                         np.linspace(-ax_lim, ax_lim, 10))
    normal = np.cross(u, v)
    d = -u @ normal
    z = (-normal[0]*xx - normal[1]*yy - d) * 1./normal[2]
    
    if ax is None:
        plt3d = plt.figure().gca(projection='3d')
        plane = plt3d.plot_surface(xx, yy, z, alpha=alpha)
    else:
        plane = ax.plot_surface(xx, yy, z, alpha=alpha)
        
    return plane

# make movie of videos
# frames should be size (pix_x * pix_y, t)
def make_movie(frames, h, w, fig, ax, fn='videos/test.mp4'):
    im = ax.imshow(frames[:, 0].reshape((w, h)))

    def animate(i):
        frame = frames[:, i].reshape((h, w))
        im.set_array(frame)
        ax.set(title='frame:\t{}'.format(i))
        return im,

    movie = animation.FuncAnimation(fig, animate, 
                            frames=frames.shape[1], interval=33)
    movie.save(fn, fps=20)
