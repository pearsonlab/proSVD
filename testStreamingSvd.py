#!/home/pritha/anaconda3/bin/python

from scipy import signal
import numpy as np
import streamingSvd as svd
import random
import os.path

def generateTimeSeriesData():
    n = 1000
    t = np.linspace(0, 1, 500, endpoint=False)
    y_square = signal.square(2 * np.pi * 5 * t)
    y_sin = np.sin(t)
    y_saw = signal.sawtooth(2 * np.pi * 5 * t)
    data = np.zeros((0, 500)) 
    data = np.append([y_square], [y_sin], axis=0)
    data = np.append(data, [y_saw], axis=0)
    return data

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

def generateARdata(num_rows, n):
    #n = 1000
    #a = 0.6
    #num_rows = 100
    np.random.seed(1)
    A = np.zeros((0,n))
    for num in range(num_rows):
        a = random.uniform(0.9,1) 
        x = w = np.random.normal(size=n)
        for t in range(n):
            x[t] = a*x[t-1] + w[t]
        A = np.append(A, [x], axis=0)
    return A


def main():
    A = generateTimeSeriesData()
    T = svd.getSvd(A, 3, 5, 5, 1000)
    print ("Calculated SVD U")
    print (T)
    U, S, V = np.linalg.svd(A, full_matrices=False)
    print ("Numpy SVD U")
    print (U)
    num_mismatch = 0
    for i in range(3):
        if (not np.allclose(T[:,i], U[:,i], 1e-1, 1e-1) and not np.allclose(T[:,i],-U[:,i],1e-1, 1e-1)):
            print ("Mismatch in %d column\n"%i)
            num_mismatch = num_mismatch + 1
    print ("Number mismatched: %d\n"%num_mismatch)


    #Check if AR.dat exists, if not create
    if not os.path.isfile('AR.dat'):
        print ("Generating and saving data")
        A = generateARdata(1000)
        np.savetxt('AR.dat', A)

    #Load AR.dat
    #A = np.loadtxt('AR.dat')

    rank = 30
    A = generateARdata(rank, 1000)
    #T = svd.getSvd(A, 100, 100, 5, 1000)
    T = svd.getSvd(A, rank, rank, 5, 1000)
    print ("Calculated SVD U")
    print (T.shape)
    U, S, V = np.linalg.svd(A[:,:], full_matrices=False)
    print ("Numpy SVD U")
    num_mismatch = 0
    for i in range(rank):
        if (not np.allclose(T[:,i], U[:,i], 1e-1, 1e-1) and not np.allclose(T[:,i],-U[:,i],1e-1, 1e-1)):
            print ("Mismatch in %d column\n"%i)
            for j in range(rank):
                if (not np.allclose(T[j,i], U[j,i], 1e-1, 1e-1) and not np.allclose(T[j,i],-U[j,i],1e-1, 1e-1)):
                    print ("Mismatch in %d row %f %f\n"%(j, T[j,i], U[j,i]))
            num_mismatch = num_mismatch + 1
    print ("Number mismatched: %d\n"%num_mismatch)

    #A = generatePieceConstData()

if __name__ == "__main__":
    main()
