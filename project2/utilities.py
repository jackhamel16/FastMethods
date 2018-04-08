import numpy as np
import numpy.linalg as lg

def pnt2idx(x, y, level):
    # Returns index, as a string in binary, of a point using Morton Coding
    # level should be the current level in the tree
    bx = format(x, '0'+str(level)+'b')
    by = format(y, '0'+str(level)+'b')

    if level != 0:
        return('01'+''.join([by[i:i+1]+bx[i:i+1] for i in range(len(bx))]))
    else:
        return('01')
    
def idx2pnt(idx):
    bx, by = '', ''
    for i, bit in enumerate(idx[2:]):
        if i % 2 == 0:
            by += bit
        else:
            bx += bit
    return(int(bx,2), int(by,2))

def binary(idx, level):
    return('0' + format(idx, '0'+str(level)+'b'))

def estimate_rank(s, eps=1e-1):
    r = 0
    for val in s:
        if val > eps:
            r += 1
    if r == 0:
        return(1)
    else:
        return(r)

def uv_decompose(B, eps):
    U,s,V = lg.svd(B, full_matrices=0)
    rank = estimate_rank(s, eps)
    Vr = np.dot(np.diag(s)[0:rank,0:rank], V[0:rank,:])
    Ur = U[:,0:rank]
    return(Ur, Vr)

def demote_rank(B, eps):
    U,s,V = lg.svd(B, full_matrices=0)
    rank = estimate_rank(s, eps)
    return(V[0:rank,0:])
    
def merge(U1, V1, U2, V2, eps, horizontal=0):
    if horizontal == 1:
        U1, V1 = np.transpose(V1), np.transpose(U1)
        U2, V2 = np.transpose(V2), np.transpose(U2)
        
    if np.size(U1) == 0:
        U12, V12 = U2, V2
    elif np.size(U2) == 0:
        U12, V12 = U1, V1
    elif (np.size(U1) == 0) and (np.size(U2) == 0):
        U12, V12 == np.array([]), np.array([])
    else:   
        V12 = demote_rank(np.vstack((V1,V2)), eps)
        
        U12_1 = np.dot(U1, np.dot(V1, np.transpose(V12)))
        U12_2 = np.dot(U2, np.dot(V2, np.transpose(V12)))
        
        U12 = np.vstack((U12_1, U12_2))
        
    if horizontal == 1:
        return(np.transpose(V12), np.transpose(U12))
    else:
        return(U12, V12)