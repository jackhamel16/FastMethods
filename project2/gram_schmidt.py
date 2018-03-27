import numpy as np
import numpy.linalg as lg

V = np.array([[1,4,2,3],[0,0,2,0],[1,2,0,7]])

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