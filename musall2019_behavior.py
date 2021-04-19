import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import h5py

import proSVD
from utils import get_streamingSVD

datadir = '/home/pgupta/musall2019_videos/' 
sess = 'reconstructed_Cam1_init.h5'
hf = h5py.File(datadir + sess, 'r')
cam1 =  np.array(hf.get('cam1')) 

w, h, t = cam1.shape

# pro SVD
k = 10
l1 = 10
pro = proSVD.proSVD(k=k, trueSVD=True, history=t-l1)
pro.initialize(cam1[:, :, :l1].reshape((w*h, l1)))
start = time.time()
for i in range(l1, t-l1):
    dat = cam1[:, :, i].reshape((w*h,1))
    pro.updateSVD(dat)
print('proSVD:\t', time.time()-start)
Qts, Ss, Qs = (pro.Qts, pro.Ss, pro.Qs)



diff = np.diff(Qs)
norm = np.linalg.norm(diff, axis=1)
plt.plot(norm)
plt.show()

# # animating
# fig, ax = plt.subplots()
# im = ax.imshow(Qs[:, 0, 0].reshape((w, h)))


# def animate(i):
#     frame = Qs[:, 0, i].reshape((w, h))
#     im.set_array(frame)
#     return im,

# anim = anim.FuncAnimation(fig, animate, 
#                          frames=Qs.shape[2], interval=33)
# anim.save('test_Qs.mp4', fps=2)
# print('done')