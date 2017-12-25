import numpy as np
import math

def solve(M, n):
    # print('simplex begin')
    # print(M)
    i = 0
    PIV = []
    while True:
        flag = True
        for ii in range(i, M.shape[0]-1):
            if M[ii, -1] < 0:
                flag = False
                break
            else:
                i = ii
        if flag:
            break # all b_i s are positive
        
        # b_ii is the first negative b_i == b_k
        FEASIBLE = False
        for jj in range(0, M.shape[1]-1):
            if M[ii, jj] < 0:
                FEASIBLE = True
                break

        if not FEASIBLE:
            return None, None, None

        # pivot is in column jj
        piv_col = jj
        piv_row = ii
        MIN = M[ii, -1]/M[ii, piv_col]
        for iii in range(M.shape[0]-1):
            if M[iii, -1] >= 0 and M[iii, piv_col] > 0:
                tmp = M[iii, -1]/M[iii, piv_col]
                if tmp < MIN:
                    piv_row = iii
                    tmp = MIN

        #print(piv_row, piv_col)
        # take pivot about piv_row, piv_col
        PIV.append((piv_row, piv_col))
        M = pivot(M, piv_row, piv_col)
        # print(M, piv_row, piv_col)

    # all b_i s are positive
    #print('bi done')

    j = 0
    while True:
        flag = True
        for jj in range(j, M.shape[1]-1):
            if M[-1, jj] < 0:
                flag = False
                break
            else:
                j = jj
        if flag:
            break # all c_i s are positive
        
        piv_col = jj
        piv_row = -1
        MIN = np.inf
        FEASIBLE = False
        for iii in range(0, M.shape[0]-1):
            if M[iii, piv_col] > 0:
                FEASIBLE = True
                tmp = M[iii, -1]/M[iii, piv_col]
                if tmp < MIN:
                    piv_row = iii
                    MIN = tmp

        if not FEASIBLE:
            return None, None, None

        # take pivot about piv_row, piv_col
        PIV.append((piv_row, piv_col))
        M = pivot(M, piv_row, piv_col)
        # print(M, piv_row, piv_col)

    for (i, j) in PIV[::-1]:
        tmp = M[i, -1]
        M[i, -1] = M[-1, j]
        M[-1, j] = tmp

    X = np.copy(M[-1, :])[0:-1]
    Y = np.copy(M[:, -1])[0:-1]
    
    V = M[-1, -1]

    # print('simplex end')
    return X,Y,V 

def pivot(M, piv_row, piv_col):

    X = np.zeros(M.shape)
    piv = M[piv_row, piv_col]
 
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if i == piv_row and j == piv_col:
                X[i, j] = 1/piv

            elif i == piv_row:
                X[i, j] = M[i, j]/piv

            elif j == piv_col:
                X[i, j] = -1*M[i, j]/piv

            else:
                X[i, j] = M[i, j] - M[piv_row, j]*M[i, piv_col]/piv

    return X
                
                
# M = np.array([[-3/2, 11/2, 7/2],
#               [1/2, -3/2, 1/2],
#               [0, 3, 1]])

# M = np.array([[0, 1, 2, 3],
#               [-1, 0, 3, 2],
#               [2, 1, 1, 1],
#               [-1, -1, -2, 0]])

# M = np.array([[-2, -1, -2, -1],
#               [2, 3, 5, 2],
#               [-3, 4, -1, 1],
#               [1, 2, 3, 0]])

# M = np.array([[2, 1, -7, 3],
#               [-1, 0, 4, -1],
#               [1, 2, -6, 2],
#               [1, -2, -1, 0]])

# M = np.array([[-3, 3, 1, 3],
#              [2, -1, -2, 1],
#              [-1, 0, 1, 1],
#              [1, 1, -2, 0]])

# X, Y, Val = solve(M)

