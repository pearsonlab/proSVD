import numpy as np
import matplotlib.pyplot as plt

# dumb streaming svd or window svd
def get_streamingSVD(X, k, l1, l, num_iters, window=True):
    Us = np.zeros((X.shape[0], k, num_iters))
    Ss = np.zeros((k, num_iters))
    
    U, S, V_T = np.linalg.svd(X[:, :l1], full_matrices=False)
    Us[:, :, 0] = U[:, :k]
    
    t = l1
    if window:
        start = t
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