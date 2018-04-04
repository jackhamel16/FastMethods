import numpy as np
import numpy.linalg as lg
import tree
import interaction
import utilities as utils

def estimate_rank(s, eps=1e-6):
    r = 0
    value = s[0]
    while (eps < value) and (r < (np.size(s)-1)):
        r += 1
        value = s[r]
    return(r)

def uv_decompose(B):
    U,s,V = lg.svd(B, full_matrices=0)
    rank = estimate_rank(s)
#    if rank <= np.size(B[:,0]/2):
    V = np.dot(np.diag(s), V)
    Ur, Vr = U[:,0:rank], V[0:rank,:]
    return(Ur, Vr)
#    else:
#        return(0, 0, 0)
    
def demote_rank(B):
    U,s,V = lg.svd(B, full_matrices=0)
    rank = estimate_rank(s)
    return(V[0:rank,0:])
    
def merge(U1, V1, U2, V2, horizontal=0):
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
        V12 = np.transpose(run_gram_schmidt(\
                                   np.transpose(np.vstack((V1,V2)))))#[0:rank,:]
    
        U12_1 = np.dot(U1, np.dot(V1, np.transpose(V12)))
        U12_2 = np.dot(U2, np.dot(V2, np.transpose(V12)))
        
        U12 = np.vstack((U12_1, U12_2))
        
    if horizontal == 1:
        return(np.transpose(V12), np.transpose(U12))
    else:
        return(U12, V12)
        
def merge2(U1, V1, U2, V2, horizontal=0):
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
        V12 = demote_rank(np.vstack((V1,V2)))
        
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
    
tree = my_tree.tree
obs_box_idx = 16
src_box_idx = 31
G = interactions.build_G(tree[obs_box_idx], tree[src_box_idx])
G1, G2 = G[0:4,:], G[4:,:]
U1, V1 = uv_decompose(G1)
U2, V2 = uv_decompose(G2)

Ug,Vg = merge(U1,V1,U2,V2)
Ut,Vt = merge2(U1,V1,U2,V2)

Bt = np.dot(Ut,Vt)


error = (lg.norm(Bt) - lg.norm(G)) / lg.norm(G)
print(error)














