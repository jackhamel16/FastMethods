import numpy as np
import numpy.linalg as lg
import random

def estimate_rank(s, eps=1e-1):
    r = 0
    value = s[0]
    while eps < value:
        r += 1
        value = s[r]
    return(r)

def uv_decompose(B):
    U,s,V = lg.svd(B, full_matrices=0)
    r = estimate_rank(s)
    if r <= np.size(B[:,0]):
        V = np.dot(np.diag(s), V)
        Ur, Vr = U[:,0:r], V[0:r,:]
        return(Ur, Vr, r)
    else:
        return(0, 0)
        
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
        
obs_srcs = my_tree.tree[16]
src_srcs = my_tree.tree[17]

B = interactions.build_G(obs_srcs, src_srcs)

# y are cols and x are rows
ydim, xdim = len(src_srcs), len(obs_srcs)
n = xdim

print('obs count: ', xdim, 'src count: ', ydim)
x = np.random.randint(10, size=ydim)

lox, hix = int(np.floor(xdim/2)), int(np.ceil(xdim/2))
loy, hiy = int(np.floor(ydim/2)), int(np.ceil(ydim/2))
B1, B2 = B[0:lox,0:loy], B[lox:xdim+1,0:loy]
B3, B4 = B[0:lox,loy:ydim+1], B[lox:xdim+1,loy:ydim+1]

B_list = [B1, B2, B3, B4]
UV_list = [uv_decompose(B) for B in B_list]
UV_lr_list = [(U[:,0:r],V[0:r,:]) for (U,V,r) in UV_list]

(U1, V1), (U2, V2) = UV_lr_list[0], UV_lr_list[1]
V1_gs = np.transpose(run_gram_schmidt(np.transpose(V1)))
V2_gs = np.transpose(run_gram_schmidt(np.transpose(V2)))

V12 = np.vstack((V1_gs, V2_gs)) #You were having issues with dimensions of htese amtrices
#U12 = np.vstack((np.dot(np.dot(U1, V1), np.transpose(V12))))

#U,V = uv_decompose(B)
#if (U == 0) and (V == 0):
#    prod = np.dot(B, x)
#else:
#    B2 = np.dot(U, V)
#    prod2 = np.dot(B2, x)

#error = (lg.norm(prod2) - lg.norm(prod))/ lg.norm(prod)
#print('error: ', error)


