#!/home/pritha/anaconda3/bin/python

from scipy import signal
import numpy as np
import streamingSvd as svd
import os.path
import scipy.io as spio
#import hbt_auth
#import hbt_dataset

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

def getData():
    cols = 100
    A = np.zeros((0, cols))
    return A

#Compare two SVD-s
def compareSvds(T, U, rank, S):
    num_mismatch = 0
    for i in range(rank):
        if (S[i] > 1e-5 and not np.allclose(T[:,i], U[:,i], 1e-1, 1e-1) and not np.allclose(T[:,i],-U[:,i],1e-1, 1e-1)):
            print ("Mismatch in %d column S_val:%d\n"%(i,S[i]))
            #print (T[:,i])
            #print (U[:,i])
            num_mismatch = num_mismatch + 1
    return num_mismatch


def readMatFile(fileName):
    # Read .mat file
    mat = spio.loadmat(fileName)

    #Get main key from mat file
    keys = list(mat.keys())
    keys.remove('__header__')
    keys.remove('__globals__')
    keys.remove('__version__')
    main_key = keys[0]

    #Access all elements in array
    data = mat[main_key][0][0][0]
    data_len = np.asscalar(mat[main_key][0][0][1][0])
    freq = np.asscalar(mat[main_key][0][0][2][0])
    channel = np.ravel(mat[main_key][0][0][3][0])
    seq =  np.asscalar(mat[main_key][0][0][4][0])
    return (main_key, data, data_len, freq, channel, seq)

def testTimeSeriesData():
    A = generateTimeSeriesData()
    T = svd.getSvd(A, 3, 5, 5, 1)
    U, S, V = np.linalg.svd(A[:,:5], full_matrices=False)
    num_mismatch = compareSvds(T, U, 3, S)
    print ("Number mismatched:%d"%num_mismatch)

def testARData():
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
        num_mismatch = compareSvds(T, U, rank, S)
        print ("%d Number of columns: %d Number mismatched: %d\n"%(j,max_num,num_mismatch))
        if (num_mismatch != 0):
            exit()

def testEegData():
    main_key, data, data_len, freq, channel, seq = readMatFile('data/Dog_1/Dog_1_preictal_segment_0003.mat')
    print ("main_key:%s Seq:%d, Freq:%d Hz, data_len:%d num_electrodes:%d,%d num_columns:%d"%(main_key, seq, freq, data_len,channel.shape[0], data.shape[0], data.shape[1]))
    A = data
    rank = data.shape[0]
    num_cols = data.shape[1]
    per_iter = 5
    max_iter= int(data.shape[1]/per_iter)
    print (max_iter)

    #Calculate online SVD
    j = max_iter+1
    T = svd.getSvd(A, rank, rank, per_iter, j)

    #Calculate full SVD of the number of columns as covered by online algorithm
    max_num = rank+5*j
    if (max_num % num_cols == 0):
        if (max_num > num_cols):
            max_num = num_cols
        U, S, V = np.linalg.svd(A[:,:max_num], full_matrices=False)
    else:
        if (max_num > num_cols):
            t = max_num % num_cols
            #If number of columns more than max, then augment the extra columns at the end of A
            aug_A = np.append(A, A[:, :t], axis=1)
        else:
            #If number of columns less than max, use A till the number of columns
            aug_A = A[:, :max_num]
        U, S, V = np.linalg.svd(aug_A[:,:], full_matrices=False)
    num_mismatch = compareSvds(T, U, rank, S)
    print ("%d Number of columns: %d Number mismatched: %d\n"%(j,max_num,num_mismatch))
    print (S)


    #for j in range(max_num):
    #    max_num = rank+5*j
    #    T = svd.getSvd(A, rank, rank, per_iter, j)
    #    if (max_num % num_cols == 0):
    #        if (max_num > num_cols):
    #            max_num = num_cols
    #        U, S, V = np.linalg.svd(A[:,:max_num], full_matrices=False)
    #    else:
    #        if (max_num > num_cols):
    #            t = max_num % num_cols
    #            #If number of columns more than max, then augment the extra columns at the end of A
    #            aug_A = np.append(A, A[:, :t], axis=1)
    #        else:
    #            #If number of columns less than max, use A till the number of columns
    #            aug_A = A[:, :max_num]
    #        U, S, V = np.linalg.svd(aug_A[:,:], full_matrices=False)
    #    num_mismatch = compareSvds(T, U, rank)
    #    if (j % 25 == 0):
    #        print ("%d Number of columns: %d Number mismatched: %d\n"%(j,max_num,num_mismatch))
    #    if (num_mismatch != 0):
    #        print ("%d Number of columns: %d Number mismatched: %d\n"%(j,max_num,num_mismatch))
    #        exit()

def testEegData1(s, len1, len2):
    x = np.loadtxt(s)
    print (x.shape)
    data = np.transpose(x)

    A = data
    rank = data.shape[0]
    num_cols = data.shape[1]
    per_iter = 5
    max_iter= int((data.shape[1]-rank)/per_iter)
    print (max_iter)


    U, S, V = np.linalg.svd(A[:,:len1], full_matrices=False)
    U1, S1, V1 = np.linalg.svd(A[:,len1:len2], full_matrices=False)

    """

    j = max_iter+1
    #for j in range(100, max_iter, 1000):
    for j in range(0, max_iter+1, 100):
        #Calculate online SVD
        T = svd.getSvd(A, rank, rank, per_iter, j)

        #Calculate full SVD of the number of columns as covered by online algorithm
        max_num = rank+5*j
        if (max_num % num_cols == 0):
            if (max_num > num_cols):
                max_num = num_cols
            U, S, V = np.linalg.svd(A[:,:max_num], full_matrices=False)
        else:
            if (max_num > num_cols):
                t = max_num % num_cols
                #If number of columns more than max, then augment the extra columns at the end of A
                aug_A = np.append(A, A[:, :t], axis=1)
            else:
                #If number of columns less than max, use A till the number of columns
                aug_A = A[:, :max_num]
            U, S, V = np.linalg.svd(aug_A[:,:], full_matrices=False)
        num_mismatch = compareSvds(T, U, rank, S)
        print ("%s %d Number of columns: %d Number mismatched: %d aug_A_cols:%d\n"%(s, j,max_num,num_mismatch,aug_A.shape[1]))
    """


def getWindowSvd(s, len1, len2, iter_cols):
    x = np.loadtxt(s)
    print (x.shape)
    data = np.transpose(x)

    A = data
    rank = data.shape[0]
    num_cols = data.shape[1]
    per_iter = 5
    max_iter= int((data.shape[1]-rank)/per_iter)
    print (max_iter)

    S_list = np.zeros((rank,1))

    #Get SVD of every iter_cols
    for j in range(0, num_cols-iter_cols, iter_cols):
        begin = j
        end = j + iter_cols - 1
        if (end > num_cols):
            break
        U, S, V = np.linalg.svd(A[:,begin:end], full_matrices=False)
        S_arr = np.ndarray((rank,1), buffer=S)
        S_list = np.append(S_list, S_arr, axis=1)

    print (S_list.shape)

    return S_list



def main():
    #testTimeSeriesData()
    #testARData()
    #testEegData()
    #testEegData1("/hdd/pritha_data/EDF_data/chb01_03.dat", 766976, 777216)
    #testEegData1("/hdd/pritha_data/EDF_data/chb01_04.dat", 375552, 382464)
    #testEegData1("/hdd/pritha_data/EDF_data/chb01_15.dat", 443392, 453632)
    #testEegData1("/hdd/pritha_data/EDF_data/chb02_19.dat", 862464, 864768)

    S_list = getWindowSvd("/hdd/pritha_data/EDF_data/chb02_19.dat", 862464, 864768, 1000)


if __name__ == "__main__":
    main()
