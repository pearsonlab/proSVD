import numpy as np
import matplotlib.pyplot as plt
import proSVD
from utils import get_streamingSVD

 
def test_proSVD():
    n, t = 20, 100
    t_init = 15

    A = np.random.uniform(size=(n, t))

    k = 15
    l1 = k
    l = 20
    num_iters = 10 #int(np.ceil(t - l1) / l)

    # true svd
    Us, Ss = get_streamingSVD(A, k, l1, l, num_iters, window=False)

    # procrustean SVD
    A_init = A[:, :t_init]
    pro = proSVD.proSVD(k=15, history=A.shape[1]-l1, trueSVD=True)
    pro.initialize(A_init)
    for i in np.arange(l1, l1+num_iters):
        dat = A[:, i:i+1]
        pro.updateSVD(dat)
    Qts, Ss, Qs = (pro.Qts, pro.Ss, pro.Qs)

    W = pro.get_W(A_init)
    print(W)

def main():
    test_proSVD()

if __name__ == "__main__":
    main()
