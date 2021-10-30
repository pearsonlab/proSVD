# TODO: add preprocessing functions to proSVD? 
#       demeaning, normalizing, filters, etc.
#       add 2 function inputs: preprocessing functions and post update

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
    # track_diff   bool    - whether you want to keep a diff 
    def __init__(self, k, w_len=1, w_shift=None, decay_alpha=1, trueSVD=False, history=0, track_diff=True):
        self.k = k
        self.w_len = w_len
        self.w_shift = w_shift if w_shift else w_len # defauls to nonoverlapping chunks of w_len cols
        self.decay_alpha = decay_alpha
        self.trueSVD = trueSVD
        self.history = history
        self.track_diff = track_diff
        self.proj_mean = np.zeros((k))  # for global mean of projected data (to get projected variance)

    
    def initialize(self, A_init, Q_init=None, B_init=None):
        ## Ainit just for initialization, so l1 is A.shape[1]
        n, l1 = A_init.shape

        ## make sure A_init.shape[1] >= k
        assert l1 >= self.k, "please init with # of cols >= k"

        self.global_mean = A_init.mean(axis=1) # for global mean of observed data (for demeaning before projecting)

        # initialize Q and B from QR of A_init, W as I
        Q, B = np.linalg.qr(A_init, mode='reduced')
        ## TODO: other init strategies?
        self.Q = Q[:, :self.k] if Q_init is None else Q_init
        self.B = B[:self.k, :l1] if B_init is None else B_init

        # self.W = np.eye(l1) # TODO: figure out if we want W
        if self.trueSVD:
            U_init, S_init, _ = np.linalg.svd(A_init, full_matrices=False)
            self.U = U_init[:, :self.k]
            self.S = S_init[:self.k]

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
                self.Us[:, :, 0] = self.U
                self.Ss = np.zeros((self.k, self.history+1))
                self.Ss[:, 0] = self.S
        self.t = 1 
        
    # method to do common run through data (replaces pro.updateSVD())
    # initializes and updates proSVD
    def run(self, A, num_init, num_iters=None, ref_basis=None):
        n, T = A.shape
        A_init = A[:, :num_init]
        self.initialize(A_init)

        if num_iters is None: # do iters to go through once
            num_iters = np.floor((A.shape[1] / self.w_shift) - (self.w_len / self.w_shift)).astype('int')
        update_times = np.arange(1, num_iters) * self.w_shift # index of when updates happen
        update_times += num_init
        
        # for svd and prosvd projections, variance explained
        projs = [np.zeros((self.k, A.shape[1]-num_init)) for i in range(2)]  # subtract l1 - init proj
        frac_vars = [np.zeros(projs[i].shape) for i in range(2)]
        # derivatives
        derivs = np.zeros((self.k, num_iters))

        # run proSVD online
        for i, t in enumerate(update_times): 
            dat = A[:, t:t+self.w_len]

            if self.track_diff:
                Q_prev = self.Q
            # ------ Update ------ #
            self._updateSVD(dat, ref_basis)
            # -------------------- #
            if self.track_diff:
                self.curr_diff = Q_prev - self.Q

            if self.history:
                self.Qs[:, :, self.t] = self.Q
                # self.Ws[:, :, self.t] = self.W
                if self.trueSVD:
                    self.Us[:, :, self.t] = self.U
                    self.Ss[:, self.t] = self.S
                    # self.Wts[:, :, self.t] = self.Wt
            self.t += 1

            # getting proj and variance explained
            for j, basis in enumerate([self.U, self.Q]):
                projs[j][:, t:t+self.w_len] = basis.T @ dat
                curr_proj_vars = projs[j][:, :t-num_init].var(axis=1)[:, np.newaxis]
                total_vars = A[:, :t].var(axis=1)
                frac_vars[j][:, t:t+self.w_len] = curr_proj_vars / total_vars.sum()
            # proSVD basis derivatives
            derivs[:, i] = np.linalg.norm(self.curr_diff, axis=0)

        return projs, frac_vars, derivs
        
    # internal func to do a single iter of basis update given some data A
    def _updateSVD(self, A, ref_basis=None):
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
        # solution for a 'reference' basis
        if ref_basis is not None:
            Mu = ref_basis.T @ Q_hat @ U[:, :self.k]
        else: # solution for 
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

        # simpler way of getting Tv
        V1 = (V.T)[:,0:self.k]
        _, Tv = rq(V1) 

        ## UPDATING Q, B
        Gu_1 = U[:, :self.k] @ Tu.T
        # Gv_1 = V[:, :self.k] @ Tv.T
        self.Q = Q_hat @ Gu_1
        self.B = Tu @ np.diag(diag[:self.k]) @ Tv.T
        # self.W = W_hat @ Gv_1
    
        # Getting true SVD basis
        if self.trueSVD:
            U, S, V = np.linalg.svd(self.B, full_matrices=False)
            self.U = self.Q @ U
            self.S = S
            # self.Wt = self.W @ V


   # update the SVD with some given data (DEPRECATED)
    # A should be in shape (n, t) (getting new colums of data)
    def updateSVD(self, A, ref_basis=None, chunk_size=0):
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

            if self.track_diff:
                Q_prev = self.Q
            # ------ Update ------ #
            self._updateSVD(A_plus, ref_basis)
            # -------------------- #
            if self.track_diff:
                self.curr_diff = Q_prev - self.Q

            if self.history:
                self.Qs[:, :, self.t] = self.Q
                # self.Ws[:, :, self.t] = self.W
                if self.trueSVD:
                    self.Us[:, :, self.t] = self.U
                    self.Ss[:, self.t] = self.S
                    # self.Wts[:, :, self.t] = self.Wt

            self.t += 1