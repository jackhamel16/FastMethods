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
    value = s[0]
    while eps < value:
        r += 1
        value = s[r]
    return(r)

def uv_decompose(B, r):
    U,s,V = lg.svd(B, full_matrices=0)
#    r = estimate_rank(s)
    if r <= np.size(B[:,0]/2):
        V = np.dot(np.diag(s), V)
        Ur, Vr = U[:,0:r], V[0:r,:]
        return(Ur, Vr)
    else:
        return(0, 0, 0)
    
def merge(U1, V1, U2, V2, horizontal=0):
    if horizontal == 1:
        U1, V1 = np.transpose(V1), np.transpose(U1)
        U2, V2 = np.transpose(V2), np.transpose(U2)
        
    V12 = np.transpose(run_gram_schmidt(np.transpose(np.vstack((V1,V2)))))

    U12_1 = np.dot(U1, np.dot(V1, np.transpose(V12)))
    U12_2 = np.dot(U2, np.dot(V2, np.transpose(V12)))
    
    U12 = np.vstack((U12_1, U12_2))
    
    if horizontal == 1:
        return(np.transpose(V12), np.transpose(U12))
    else:
        return(U12, V12)

def run_gram_schmidt(V):
    n = np.size(V[:,0])
    k = np.size(V[0,:])
    U = np.array([np.zeros(k) for i in range(n)])
    U[:,0] = V[:,0] / np.sqrt(np.dot(V[:,0], V[:,0]))
    for i in range(1,k):
        U[:,i] = V[:,i]
        for j in range(i):
            U[:,i] = U[:,i] - np.dot(U[:,i], U[:,j]) / \
                              np.dot(U[:,j], U[:,j]) * U[:,j]
        U[:,i] = U[:,i] / np.sqrt(np.dot(U[:,i], U[:,i]))
    return(U)