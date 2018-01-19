#!/home/pritha/anaconda3/bin/python

import numpy as np

def getSvd(A, k, l, num_iter):
    s = A.shape[1]
    A_init = A[:, 0:k]

    Q, R = np.linalg.qr(A_init, mode='reduced')

    t = k
    
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

        #Orthogonal Procrustes singular basis
        M = R_T.dot(G1_T.dot(R_hat))
        V1 = V[:, 0:k]
        M = M.dot(V1)
        
        #Find U_tilda, V_tilda from SVD of M
        U_tilda, diag_tilda, V_tilda_T = np.linalg.svd(M, full_matrices=False)
        
        #Find T as product of U_tilda, V_tilda
        T = U_tilda.dot(V_tilda_T)
        
        #Calculate new R of this iteration using T
        #R = G_u_Transpose * R_hat * V1 * Tv_transpose
        T_trans = np.transpose(T)
        Gv1 = V1.dot(T_trans)
        R = G1_T.dot(R_hat.dot(Gv1))

    U, S, V = np.linalg.svd(R, full_matrices=False)
    Q = Q.dot(U)

    return Q

