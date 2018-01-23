#!/home/pritha/anaconda3/bin/python

from scipy import signal
import numpy as np
import streamingSvd as svd

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

def main():
    A = generateTimeSeriesData()
    T = svd.getSvd(A, 3, 5, 5, 1000)
    print ("Calculated SVD U")
    print (T)
    U, S, V = np.linalg.svd(A, full_matrices=False)
    print ("Numpy SVD U")
    print (U)

    #A = generatePieceConstData()

if __name__ == "__main__":
    main()
