#!/home/pritha/anaconda3/bin/python

from scipy import signal
import numpy as np
import streamingSvd as svd
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
    np.random.seed(1)
    A = np.zeros((0,n))
    for num in range(num_rows):
        a = np.random.uniform(0.9,1) 
        x = w = np.random.normal(size=n)
        for t in range(n):
            x[t] = a*x[t-1] + w[t]
        A = np.append(A, [x], axis=0)
    return A

def generateFakePatientData(rows, cols):
    t = np.linspace(0, 1, cols, endpoint=False)
    y_square = signal.square(2 * np.pi * 5 * t)
    y_saw = signal.sawtooth(2 * np.pi * 5 * t)
    t = np.linspace(0, 20*np.pi, cols, endpoint=False)
    y_sin = np.sin(t)
    A = np.zeros((0, cols))
    np.random.seed(1)
    for i in range(0, rows):
        a = np.random.uniform(0.9,3) 
        data = a[0]*y_square + a[1]*y_sin + a[2]*y_saw
        A = np.append(A, [data], axis=0)
    return A

#Compare two SVD-s
def compareSvds(T, U, rank):
    num_mismatch = 0
    for i in range(rank):
        if (not np.allclose(T[:,i], U[:,i], 1e-1, 1e-1) and not np.allclose(T[:,i],-U[:,i],1e-1, 1e-1)):
            #print ("Mismatch in %d column\n"%i)
            num_mismatch = num_mismatch + 1
    return num_mismatch

def main():
    A = generateTimeSeriesData()
    T = svd.getSvd(A, 3, 5, 5, 1)
    U, S, V = np.linalg.svd(A[:,:5], full_matrices=False)
    num_mismatch = compareSvds(T, U, 3)
    print ("Number mismatched:%d"%num_mismatch)


    #Check if AR.dat exists, if not create
    #if not os.path.isfile('AR.dat'):
    #    print ("Generating and saving data")
    #    A = generateARdata(1000)
    #    np.savetxt('AR.dat', A)

    #Load AR.dat
    #A = np.loadtxt('AR.dat')

    rank = 100
    A = generateARdata(rank, 1000)
    for j in range(200):
        max_num = rank+5*j
        T = svd.getSvd(A, rank, rank, 5, j)
        if (max_num % 1000 == 0):
            if (max_num > 1000):
                max_num = 1000
            U, S, V = np.linalg.svd(A[:,:max_num], full_matrices=False)
        else:
            if (max_num > 1000):
                t = max_num % 1000
                #If number of columns more than max, then augment the extra columns at the end of A
                aug_A = np.append(A, A[:, :t], axis=1)
            else:
                #If number of columns less than max, use A till the number of columns
                aug_A = A[:, :max_num]
            U, S, V = np.linalg.svd(aug_A[:,:], full_matrices=False)
        num_mismatch = compareSvds(T, U, rank)
        print ("%d Number of columns: %d Number mismatched: %d\n"%(j,max_num,num_mismatch))
        if (num_mismatch != 0):
            exit()


if __name__ == "__main__":
    main()
