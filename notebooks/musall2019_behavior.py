#%%
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import h5py

import proSVD
from utils import get_streamingSVD, make_movie
#%%
datadir = '/home/pgupta/mouse_videos/musall2019/' 
# sess = 'reconstructed_Cam1_init.h5'
# sess = 'reconstructed_Cam1_trial_int2_3.h5'
sess = 'reconstructed_Cam1_trial5.h5'
hf = h5py.File(datadir + sess, 'r')
cam1 =  np.array(hf.get('cam1')) 

w, h, t = cam1.shape
all_frames = cam1.reshape((w*h, t))

#%%
# regular SVD
# U, S, V = np.linalg.svd(all_frames, full_matrices=False)

#%%
# pro SVD
k = 10
l1 = k
pro = proSVD.proSVD(k=k, trueSVD=False, history=0)
pro.initialize(all_frames[:, :l1])

norm = np.zeros((t-l1, k))
start = time.time()

for i, curr_t in enumerate(range(l1, t)):
    prev_Q = pro.Q
    frame = all_frames[:, curr_t, np.newaxis]
    pro.updateSVD(frame)
    curr_Q = pro.Q
    norm[i, :] = np.linalg.norm(prev_Q - curr_Q, axis=0)

print('proSVD:\t', time.time()-start)

#%%
plt.plot(norm)
plt.show()


#%%
# animating
fig, ax = plt.subplots()
# make_movie(all_frames, fig, ax, fn='videos/musall_real.mp4')
make_movie(Qs[:, 1, :], w, h, fig, ax, fn='videos/musall_Qs.mp4')
# print('done')
# %%
