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

n = 3
x = np.linspace(-3, 2, 6)
#y1 = np.piecewise(x, [x < 0, x >=0], [0, 1])
#y2 = np.piecewise(x, [x < 0, x >=0], [1, 0])
y1 = np.array([0, 1])
y2 = np.array([1, 0])
A = np.zeros((0, 2))
for i in range(0, n):
    a = np.random.randint(-100, 101, size=2)
    data = a[0]*y1 + a[1]*y2
    A = np.append(A, [data], axis=0)
A = np.append(A, [y1], axis=0)
A = np.append(A, [y2], axis=0)
A = np.transpose(A)
print (A.shape)
print (A)

#Get back singular vectors
Q, R = np.linalg.qr(A, mode='reduced')
print (Q)

s = 30
k = 10
l = 5

A = np.matrix('1, 2, 0; 2, 1, 0')
for i in range(k, s, l):
    print (A[:, 2:3])
    print (i)
    print (i+l-1)
