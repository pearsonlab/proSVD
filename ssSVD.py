# Function get_ssSVD takes a total data matrix A and returns the computed stable bases over time

# Inputs:
# A - full data matrix, shape (features, time)
# k - number of basis vectors kept (reduced dimension)
# l1 - number of columns to initiate with (helpful if this is larger than k)
# l - number of columns processed at a time (chunk size)
# num_iter - number of times to iterate through data matrix 
#            (looping multiple times through same matrix is supported)
# decay_alpha - power to which singular values are raised. implements forgetting for decay
# silent - if false, prints sizes of matrices 

# Outputs:
# Qtcoll - shape (n, min(l, k), num_iter) tensor of all true SVD bases found over num_iters (can discard if true SVD not wanted)
# Scoll - shape (min(l, k), num_iters) matrix of singular values over num_iters
# Qcoll - shape (n, min(l, k), num_iter) tensor of all ssSVD bases found over num_iters

import scipy
import numpy as np

def get_ssSVD(A, k, l1, l, num_iter, decay_alpha=1, silent=True):
    n, s = A.shape
    
    dim2_size = min(l1, k)
    Qcoll = np.zeros((n, dim2_size, num_iter))
    Qtcoll = np.zeros((n, dim2_size, num_iter))
    Scoll = np.zeros((dim2_size, num_iter))
    
    A_init = A[:, 0:l1]
    Q, R = np.linalg.qr(A_init, mode='reduced')
    
    t = l1
    for i in range(0, num_iter):

        if (t >= s):
            t = t-s

        #New data A+
        #If wrapping around, concatenate last and first sections together
        if (t+l >= s):
            A_plus = A[:, t:s]
            tmp = t+l-s
            A_plus = np.append(A_plus, A[:,0:tmp], axis=1)
            t = tmp
        else:
            A_plus = A[:, t:t+l]
            t = t+l
        
        #QR decomposition of additional data
        Q_T = np.transpose(Q)
        R_T = np.transpose(R)
        C = Q_T.dot(A_plus)
        A_perp = A_plus - Q.dot(C)
        Q_perp, R_perp = np.linalg.qr(A_perp, mode='full')

        #Calculate QR decomposition of augmented data matrix, Q_hat, R_hat
        #Q_hat is simple appending of Qi-1 and Q_perp
        #R_hat is based on Figure 3.1 in Baker's thesis
        Q_hat = np.append(Q, Q_perp, axis=1)
        R_prev = np.append(R, C, axis=1)
        tmp = np.zeros((R_perp.shape[0], R.shape[1]))
        tmp = np.append(tmp, R_perp, axis=1)

        R_hat = np.append(R_prev, tmp, axis=0)
        
        #SVD of R_hat (B_hat)
        U, diag, V_T = np.linalg.svd(R_hat, full_matrices=False)
        # decaying (implements forgetting)
        diag = np.power(diag, decay_alpha)
        V = np.transpose(V_T)
        
        #Orthogonal Procrustes singular basis
        M = Q_T.dot(Q_hat) 
        U1 = U[:, 0:k]
        M = M.dot(U1)
        
        #Find U_tilda, V_tilda from SVD of M
        U_tilda, diag_tilda, V_tilda_T = np.linalg.svd(M, full_matrices=False)
        #Find T as product of U_tilda, V_tilda
        T = U_tilda.dot(V_tilda_T)
        
        #Calculate new Q of this iteration using T
        #Q = Q_hat * U1 * T_transpose
        T_trans = np.transpose(T)
        G1 = U1.dot(T_trans)
        G1_T = np.transpose(G1)
        Q = Q_hat.dot(G1)
        
        #Calculation of new R does not need Orthogonal Procrustes since
        # we do not care
        V1 = V[:,0:k]
        G1v, Tv = scipy.linalg.rq(V1)
        Tv_T = np.transpose(Tv)
        #R = G1_T.dot(R_hat.dot(G1v))
        #Simplifying this expression, we get
        #R = G1_T * U * diag(S) * V_T * V * Tv_T
        #R = G1_T * U * diag(S) * Tv_T
        #But G1_T = T * U1_T => R = T * U1_T * U1 * diag(S) * Tv_T
        #R = T * diag(S) * Tv_T
        R = T.dot(np.diag(diag[0:k]).dot(Tv_T))
        ##Orthogonal Procrustes singular basis
        #M = R_T.dot(G1_T.dot(R_hat))
        #V1 = V[:, 0:k]
        #M = M.dot(V1)
        #
        ##Find U_tilda, V_tilda from SVD of M
        #U_tilda, diag_tilda, V_tilda_T = np.linalg.svd(M, full_matrices=False)
        #
        ##Find T as product of U_tilda, V_tilda
        #T = U_tilda.dot(V_tilda_T)
        #
        ##Calculate new R of this iteration using T
        ##R = G_u_Transpose * R_hat * V1 * Tv_transpose
        #T_trans = np.transpose(T)
        #Gv1 = V1.dot(T_trans)
        #R = G1_T.dot(R_hat.dot(Gv1))
        
        # Collecting all ssSVD bases Q
        Qcoll[:, :, i] = Q[:, :k]
        
        # Rotates current basis to true SVD basis, unnecessary if true SVD not wanted 
        U, S, V = np.linalg.svd(R, full_matrices=False)
        Scoll[:, i] = S[:k]
        Qtcoll[:, :, i] = Q.dot(U)
        
    if not silent:
        print('Q_perp shape:\t' + str(Q_perp.shape))
        print('R_perp shape:\t' + str(R_perp.shape))
        print('Aplus shape:\t' + str(A_plus.shape))
        print('R_hat shape:\t' + str(R_hat.shape))
        print('U (Rhat) shape:\t' + str(U.shape))
        print('Q shape :\t' + str(Q.shape))
        print('Qhat shape :\t' + str(Q_hat.shape))
        print('U1 shape:\t' + str(U1.shape))
        print('M shape :\t' + str(M.shape))
        print('T shape :\t' + str(T.shape))
        print('G1 shape:\t' + str(G1.shape))
        print('new Q shape:\t' + str(Q.shape))
        print('R shape:\t' + str(R.shape))
            

    return Qtcoll, Scoll, Qcoll


class proSVD():

    def __init__(self, k, l, window, decay_alpha=1):
        self.k = k                          # k - number of basis vectors kept (reduced dimension)
        self.l = l                          # l - number of columns processed at a time (chunk size)
        self.window = window                # sliding window size..TBD
        self.decay_alpha = decay_alpha      # power to which singular values are raised. implements forgetting for decay

    def initialize(self, Ainit):
        ## Ainit just for initialization, so l1 is A.shape[1]
        n, l1 = Ainit.shape
        dim2_size = min(l1, self.k)

        ## these may need to be some kind of circular buffer
        self.Qcoll = np.zeros((n, dim2_size, self.window))
        self.Qtcoll = np.zeros((n, dim2_size, self.window))
        self.Scoll = np.zeros((dim2_size, self.window))
        
        self.Q, self.R = np.linalg.qr(Ainit, mode='reduced')

        self.t = 0
        
    def updateSVD(self, A):
        ## Update our basis vectors based on a chunk of new data
        ## Currently assume we get chunks as specificed in self.l
        ## TODO: use loop over chunk sizes here if requested?

        # QR decomposition of new data
        C = (self.Q.T).dot(A) 
        A_perp = A - self.Q.dot(C) 
        Q_perp, R_perp = np.linalg.qr(A_perp, mode='full') ##NOTE: 'full' depreicated, alias of reduced (?)

        # Calculate QR decomposition of augmented data matrix, Q_hat, R_hat
        # Q_hat is simple appending of Qi-1 and Q_perp
        # R_hat is based on Figure 3.1 in Baker's thesis
        Q_hat = np.concatenate((self.Q, Q_perp), axis=1) ##NOTE: append-->concat
        R_prev = np.concatenate((self.R, C), axis=1)
        tmp = np.zeros((R_perp.shape[0], self.R.shape[1]))
        tmp = np.concatenate((tmp, R_perp), axis=1)

        R_hat = np.concatenate((R_prev, tmp), axis=0)
        
        # SVD of R_hat (B_hat)
        U, diag, V = np.linalg.svd(R_hat, full_matrices=False)
        # decaying (implements forgetting)
        diag = np.power(diag, self.decay_alpha)
        
        # Orthogonal Procrustes singular basis
        M = (self.Q.T).dot(Q_hat).dot(U[:,0:self.k]) 
        
        # Find U_tilda, V_tilda from SVD of M
        U_tilda, _, V_tilda_T = np.linalg.svd(M, full_matrices=False)
        # Find T as product of U_tilda, V_tilda
        T = U_tilda.dot(V_tilda_T)
        
        # Calculate new Q of this iteration using T
        G1 = (U[:, 0:self.k]).dot(T.T)
        self.Q = Q_hat.dot(G1)
        
        # Calculation of new R does not need Orthogonal Procrustes since
        # we do not care
        V1 = (V.T)[:,0:self.k]
        _, Tv = scipy.linalg.rq(V1) 
        self.R = T.dot(np.diag(diag[0:self.k]).dot(Tv.T))

        # Collecting all ssSVD bases Q
        self.Qcoll[:, :, self.t] = self.Q[:, :self.k]
        
        # Rotates current basis to true SVD basis, unnecessary if true SVD not wanted 
        U, S, V = np.linalg.svd(self.R, full_matrices=False)
        self.Scoll[:, self.t] = S[:self.k]
        self.Qtcoll[:, :, self.t] = self.Q.dot(U)

        self.t += 1