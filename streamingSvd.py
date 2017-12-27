#!/home/pritha/anaconda3/bin/python

import numpy as np


def generatePieceConstData():
    n = 10
    m = 11
    x = np.linspace(-5, 5, m)
    y1 = np.piecewise(x, [x < 0, x >=0], [0, 1])
    y2 = np.piecewise(x, [x < 0, x >=0], [1, 0])
    print (y1 * y2)
    A = np.zeros((0, m))
    A = np.append(A, [y1], axis=0)
    A = np.append(A, [y2], axis=0)
    for i in range(0, n):
        a = np.random.randint(-100, 101, size=2)
        data = a[0]*y1 + a[1]*y2
        A = np.append(A, [data], axis=0)
    A = np.transpose(A)
    return A


def getSvd(A, k, l):
    s = A.shape[1]
    A_init = A[:, 0:k]

    U, S, V = np.linalg.svd(A_init, full_matrices=False)
    S[S < 1e-10] = 0
    #print (S)
    #Original Q, R
    Q, R = np.linalg.qr(A_init, mode='reduced')
    #print (Q)
    #print (R)
    U, S, V = np.linalg.svd(R, full_matrices=False)
    S[S < 1e-10] = 0
    #print (S)

    num = 10
    t = k
    
    for i in range(0, 100):

        if (t == s):
            t = 0

        #New data A+
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
        tmp = np.zeros((R_perp.shape[0], R.shape[0]))
        tmp = np.append(tmp, R_perp, axis=1)
        R_hat = np.append(R_prev, tmp, axis=0)

        #SVD of R_hat (B_hat)
        U, diag, V = np.linalg.svd(R_hat, full_matrices=False)
        
        #Orthogonal Procrustes singular basis
        M = Q_T.dot(Q_hat) 
        U1 = U[:, 0:k]
        M = M.dot(U1)
        
        #Find U_tilda, V_tilda from SVD of M
        U_tilda, diag_tilda, V_tilda = np.linalg.svd(M, full_matrices=False)
        
        #Find T as product of U_tilda, V_tilda
        V_tilda_T = np.transpose(V_tilda)
        T = U_tilda.dot(V_tilda)
        
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
        U_tilda, diag_tilda, V_tilda = np.linalg.svd(M, full_matrices=False)
        
        #Find T as product of U_tilda, V_tilda
        T = U_tilda.dot(V_tilda)
        
        #Calculate new R of this iteration using T
        #R = G_u_Transpose * R_hat * V1 * Tv_transpose
        T_trans = np.transpose(T)
        Gv1 = V1.dot(T_trans)
        R = G1_T.dot(R_hat.dot(Gv1))


    return Q



def main():
    A = generatePieceConstData()
    T = getSvd(A, 4, 2)
    U, S, V = np.linalg.svd(T, full_matrices=False)
    #print (T.shape)
    #print (U.shape)
    U[U < 1e-10] = 0
    #print (U)
    S[S < 1e-10] = 0
    print (S)
    U, S, V = np.linalg.svd(A, full_matrices=False)
    S[S < 1e-10] = 0
    print (S)
    #print (U.shape)
    #print (V.shape)
    #print (U)
    #T_t = np.transpose(T)
    #P = T.dot(T_t)
    #w, v = np.linalg.eig(P)
    #w[w < 1e-10] = 0
    #v[v < 1e-10] = 0
    #print (w)
    #print (v)

if __name__ == "__main__":
    main()
