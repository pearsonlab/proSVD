#!/home/pritha/anaconda3/bin/python

import numpy as np

s = 10
k = 4
l = 3

#New data of this iteration
C = np.random.randn(s, l)

#Original Data
A = np.random.randn(s, k)

#Original Q, R
Q, R = np.linalg.qr(A, mode='reduced')

#A+ - Previous discomposition augmented with new data
A_plus = np.append(Q, C, axis=1)

#QR decomposition of augmented data matrix, Q_hat, R_hat
Q1, R1 = np.linalg.qr(A_plus, mode='reduced')

#SVD of R_hat (B_hat)
U, diag, V = np.linalg.svd(R1, full_matrices=False)

#Orthogonal Procrustes singular basis
Q_T = np.transpose(Q)
M = Q_T.dot(Q1) 
U1 = U[:, 0:k]
M = M.dot(U1)

#Find U_tilda, V_tilda from SVD of M
U1_1, diag_1, V1_1 = np.linalg.svd(M, full_matrices=False)

#Find T as product of U_tilda, V_tilda
V1_1_T = np.transpose(V1_1)
T = U1_1.dot(V1_1_T)

#Calculate new Q of this iteration using T
G1 = U1.dot(T)
Q_j = Q1.dot(G1)

print ("A_plus:")
print (A_plus.shape)
print ("Q1:")
print (Q1.shape)
print ("R1:")
print (R1.shape)
print ("Q_T:")
print (Q_T.shape)
print ("Q1:")
print (Q1.shape)
print ("M:")
print (M.shape)
print ("U:")
print (U.shape)
print ("U1:")
print (U1.shape)
print ("U1_1:")
print (U1_1.shape)
print ("T:")
print (T.shape)
print ("Q_j:")
print (Q_j.shape)

