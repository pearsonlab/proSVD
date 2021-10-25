# TODO: Clean up

import numpy as np
from scipy.linalg import rq

class proSVD:
    # init attributes
    # k            int     - reduced dimension
    # w_len        int     - window length
    # w_shift      int     - how many cols does the window shift by?
    # decay_alpha  float   - forgetting parameter (no memory = 0 < alpha <= 1 = no forgetting)
    # trueSVD      bool    - whether or not basis should be rotated to true SVD basis (stored as attribute U)
    # history      int     - 0 indicates no history will be kept, 
    #                       >0 indicates how many bases/singular values to keep
    def __init__(self, k, w_len=1, w_shift=1, decay_alpha=1, trueSVD=False, history=0, newMu=False):
        self.k = k
        self.w_len = w_len
        self.w_shift = w_shift
        self.decay_alpha = decay_alpha
        self.trueSVD = trueSVD
        self.history = history
        self.newMu = newMu
        self.proj_mean = np.zeros((k))  # for global mean of projected data (to get projected variance)

    
    def initialize(self, A_init):
        ## Ainit just for initialization, so l1 is A.shape[1]
        n, l1 = A_init.shape

        ## make sure A_init.shape[1] >= k
        assert l1 >= self.k, "please init with # of cols >= k"

        self.global_mean = A_init.mean(axis=1) # for global mean of observed data (for demeaning before projecting)

        # initialize Q and B from QR of A_init, W as I
        Q, B = np.linalg.qr(A_init, mode='reduced')
        ## TODO: other init strategies?
        # u, s, v = np.linalg.svd(A_init, full_matrices=False)
        # Q, B = u, np.diag(s) @ v
        
        self.Q = Q[:, :self.k]
        self.B = B[:self.k, :l1]
<<<<<<< HEAD
        # self.W = np.eye(l1) # TODO: figure out if we want W
=======
        # self.W = np.eye(l1)
>>>>>>> 39db6206fe683089720eab7f93129ac8b8f8db61

        if self.history:
            ## these may need to be some kind of circular buffer
            ## for now assuming self.history = A_full.shape[1] - l1 (if not 0)
            self.Qs = np.zeros((n, self.k, self.history+1))
            self.Qs[:, :, 0] = self.Q # init with first Q

            # this might need to be different?
            # self.Ws = np.zeros((self.k, self.w_len, self.history))

            # keeping true singular vectors/values
            if self.trueSVD:
                self.Us = np.zeros(self.Qs.shape)
                self.Us[:, :, 0] = self.Q
                self.Ss = np.zeros((self.k, self.history+1))
        self.t = 1
        
    
    # update the SVD with some given data
    # A should be in shape (n, t) (getting new colums of data)
    # optional chunk size should be > 0, indicates how many nonoverlapping cols to process
    # TODO: get rid of chunk size with w_len and w_shift
    def updateSVD(self, A, chunk_size=0):
        n, s = A.shape
        
        if chunk_size == 0:  # process all of A (as one big chunk) and update basis once
            num_iter = 1
            l = s
        else:  # process A with smaller chunks at a time, update basis multiple times
            num_iter = int(np.ceil(s / chunk_size))  # iters to go through data once
            l = chunk_size
        
        t = 0
        for i in range(num_iter):
            
            A_plus = A[:, t:t+l]
            t = t+l

            # Actual update
            self._updateSVD(A_plus)

            if self.history:
                self.Qs[:, :, self.t] = self.Q
                # self.Ws[:, :, self.t] = self.W
                if self.trueSVD:
                    self.Us[:, :, self.t] = self.U
                    self.Ss[:, self.t] = self.S
                    # self.Wts[:, :, self.t] = self.Wt

            self.t += 1
        
    
    # internal func to do a single iter of basis update given some data A
    def _updateSVD(self, A):
        _, l = A.shape
        ## Update our basis vectors based on a chunk of new data
        ## Currently assume we get chunks as specificed in self.l
        ## QR decomposition of new data
        C = self.Q.T @ A 
        A_perp = A - self.Q @ C 
        Q_perp, B_perp = np.linalg.qr(A_perp, mode='reduced')

        # Calculate QR decomposition of augmented data matrix, Q_hat, R_hat
        # Q_hat is simple appending of Qi-1 and Q_perp
        Q_hat = np.concatenate((self.Q, Q_perp), axis=1) 
        
        # R_hat is based on Figure 3.1 in Baker's thesis
        B_prev = np.concatenate((self.B, C), axis=1)
        tmp = np.zeros((B_perp.shape[0], self.B.shape[1]))
        tmp = np.concatenate((tmp, B_perp), axis=1)
        B_hat = np.concatenate((B_prev, tmp), axis=0)

        # W_hat is I_l appended as block to W
        # I_l = np.eye(l)
        # right_block = np.zeros((self.W.shape[0], l))
        # bottom_block = np.zeros((l, self.W.shape[1]))
        # W_hat = np.block([[self.W, right_block], 
        #                   [bottom_block, I_l]])
        
        ## Constructing orthogonal Gu and Gv from Tu and Tv
        # SVD of B_hat 
        U, diag, V = np.linalg.svd(B_hat, full_matrices=False)

        # decaying (implements forgetting)
        # diag = np.power(diag, self.decay_alpha)
        diag *= self.decay_alpha
        
        # Orthgonal Procrustes singular basis for Q (getting Tu)
        # Mu = self.Q.T @ Q_hat @ U[:, :self.k]
        # faster getting Mu - just the first k rows of U1??
        Mu = U[:self.k, :self.k]

        U_tilda, _, V_tilda_T = np.linalg.svd(Mu, full_matrices=False)
        Tu = U_tilda @ V_tilda_T

        # Orthogonal Procrustes singular basis for W (getting Tv)
        # TODO: W_j-1 is smaller than W_hat?
        # truncate first L rows of W_hat
        # Mv = self.W.T @ W_hat[l:, :] @ V[:, :self.k]
        # U_tilda, _, V_tilda = np.linalg.svd(Mv, full_matrices=False)
        # Tv = U_tilda @ V_tilda

<<<<<<< HEAD
        # simpler way of getting Tv
=======
        # Old way of getting Tv
>>>>>>> 39db6206fe683089720eab7f93129ac8b8f8db61
        V1 = (V.T)[:,0:self.k]
        _, Tv = rq(V1) 

        ## UPDATING Q, B
        Gu_1 = U[:, :self.k] @ Tu.T
        # Gv_1 = V[:, :self.k] @ Tv.T
        self.Q = Q_hat @ Gu_1
        self.B = Tu @ np.diag(diag[:self.k]) @ Tv.T
<<<<<<< HEAD
=======

        # Gv_1 = V[:, :self.k] @ Tv.T
>>>>>>> 39db6206fe683089720eab7f93129ac8b8f8db61
        # self.W = W_hat @ Gv_1
    
        # Getting true SVD basis
        if self.trueSVD:
            U, S, V = np.linalg.svd(self.B, full_matrices=False)
            self.U = self.Q @ U
            self.S = S
<<<<<<< HEAD
            # self.Wt = self.W @ V
=======
            # self.Wt = self.W @ V
>>>>>>> 39db6206fe683089720eab7f93129ac8b8f8db61
