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

    #Original Q, R
    Q, R = np.linalg.qr(A_init, mode='reduced')

    num = 100
    t = k
    
    for i in range(0, 100):
        if (t == s):
            t = 0
        #New data
        C = A[:, t:t+l]
        t = t+l

        #A+ - Previous discomposition augmented with new data
        A_plus = np.append(Q, C, axis=1)
        
        #QR decomposition of augmented data matrix, Q_hat, R_hat
        Q_hat, R_hat = np.linalg.qr(A_plus, mode='reduced')
        
        #SVD of R_hat (B_hat)
        U, diag, V = np.linalg.svd(R_hat, full_matrices=False)
        
        #Orthogonal Procrustes singular basis
        Q_T = np.transpose(Q)
        M = Q_T.dot(Q_hat) 
        U1 = U[:, 0:k]
        M = M.dot(U1)
        
        #Find U_tilda, V_tilda from SVD of M
        U_tilda, diag_tilda, V1_tilda = np.linalg.svd(M, full_matrices=False)
        
        #Find T as product of U_tilda, V_tilda
        V_tilda_T = np.transpose(V1_tilda)
        T = U_tilda.dot(V_tilda_T)
        
        #Calculate new Q of this iteration using T
        G1 = U1.dot(T)
        Q = Q_hat.dot(G1)
    return Q



def main():
    A = generatePieceConstData()
    T = getSvd(A, 4, 2)
    U, S, V = np.linalg.svd(T)
    print (T.shape)
    print (U.shape)

if __name__ == "__main__":
    main()
