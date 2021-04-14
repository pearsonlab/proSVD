import numpy as np
from scipy.linalg import rq

class proSVD:
    # init attributes
    # k            int     - reduced dimension
    # w_len        int     - window length
    # w_shift      int     - how many cols does the window shift by?
    # decay_alpha  float   - forgetting parameter (no memory = 0 < alpha <= 1 = no forgetting)
    # trueSVD      bool    - whether or not basis should be rotated to true SVD basis (stored as attribute Qt)
    # history      int     - 0 indicates no history will be kept, 
    #                       >0 indicates how many bases/singular values to keep
    def __init__(self, k, w_len=1, w_shift=1, decay_alpha=1, trueSVD=False, history=0):
        self.k = k
        self.w_len = w_len
        self.w_shift = w_shift
        self.decay_alpha = decay_alpha
        self.trueSVD = trueSVD
        self.history = history

    
    def initialize(self, A_init):
        ## Ainit just for initialization, so l1 is A.shape[1]
        n, l1 = A_init.shape
        dim2_size = min(l1, self.k)

        ## make sure A_init.shape[1] >= k
        assert l1 >= self.k, "please init with # of cols >= k"

        # TODO: add W history
        if self.history:
            ## these may need to be some kind of circular buffer
            ## for now assuming self.history = A_full.shape[1] - l1 (if not 0)
            self.Qs = np.zeros((n, dim2_size, self.history))
            # self.Ws = np.zeros((dim2_size, self.w_len, self.history))

            # keeping true singular vectors/values
            if self.trueSVD:
                self.Qts = np.zeros(self.Qs.shape)
                self.Ss = np.zeros((dim2_size, self.history))

        # initialize Q and B from QR of A_init, W as I
        self.Q, self.B = np.linalg.qr(A_init, mode='reduced')
        # self.W = np.eye(l1)
        self.t = 0
        
    
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

            # ACTUAL UPDATE HERE
            self._updateSVD(A_plus)

            if self.history:
                self.Qs[:, :, self.t] = self.Q
                # self.Ws[:, :, self.t] = self.W
                if self.trueSVD:
                    self.Qts[:, :, self.t] = self.Qt
                    self.Ss[:, self.t] = self.S

            self.t += 1
        
    
    # internal func to do a single iter of basis update given some data A
    def _updateSVD(self, A):
        ## Update our basis vectors based on a chunk of new data
        ## Currently assume we get chunks as specificed in self.l
        ## TODO: use loop over chunk sizes here if requested?

        # QR decomposition of new data
        C = self.Q.T @ A 
        A_perp = A - self.Q @ C 
        Q_perp, R_perp = np.linalg.qr(A_perp, mode='reduced') ##NOTE: 'full' depreicated, alias of reduced (?)

        # Calculate QR decomposition of augmented data matrix, Q_hat, R_hat
        # Q_hat is simple appending of Qi-1 and Q_perp
        # R_hat is based on Figure 3.1 in Baker's thesis
        Q_hat = np.concatenate((self.Q, Q_perp), axis=1) ##NOTE: append-->concat
        R_prev = np.concatenate((self.B, C), axis=1)
        tmp = np.zeros((R_perp.shape[0], self.B.shape[1]))
        tmp = np.concatenate((tmp, R_perp), axis=1)
        R_hat = np.concatenate((R_prev, tmp), axis=0)
        
        # SVD of R_hat (B_hat)
        U, diag, V = np.linalg.svd(R_hat, full_matrices=False)
        # decaying (implements forgetting)
        diag = np.power(diag, self.decay_alpha)
        
        # Orthogonal Procrustes singular basis
        M = self.Q.T @ Q_hat @ U[:,0:self.k]
        
        # Find U_tilda, V_tilda from SVD of M
        U_tilda, _, V_tilda_T = np.linalg.svd(M, full_matrices=False)
        # Find T as product of U_tilda, V_tilda
        T = U_tilda @ V_tilda_T
        
        # Calculate new Q of this iteration using T
        G1 = U[:, :self.k] @ T.T
        Q_full = Q_hat @ G1
        self.Q = Q_full[:, :self.k]

        # Calculation of new R does not need Orthogonal Procrustes since
        # we do not care
        V1 = (V.T)[:,0:self.k]
        _, Tv = rq(V1) 
        self.B = T @ np.diag(diag[:self.k]) @ Tv.T
    
        # Getting W and true SVD basis
        if self.trueSVD:
            # rotating to true basis
            U, S, V = np.linalg.svd(self.B, full_matrices=False)
            self.Qt = Q_full @ U
            self.S = S
            
            # getting W (basis for right singular subpsace)
            # using psuedoinv, should probably not do this
            # but since B isn't necessarily diagonal, B_inv will change more than scaling
            # TODO: do we need B_inv? can we do this more efficiently than a pseudoinv?
            # self.W = np.linalg.pinv(self.B) @ self.Q.T @ A
            
    # getting W (basis for right singular subpsace)
    # TODO: fold this into above
    def get_W(self, data):
        return np.linalg.pinv(self.B) @ self.Q.T @ data