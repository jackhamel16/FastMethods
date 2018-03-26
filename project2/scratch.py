import numpy as np
import numpy.linalg as lg
import random

obs_srcs = my_tree.tree[16]
src_srcs = my_tree.tree[]

B = interactions.build_G(obs_srcs, src_srcs)

# y are cols and x are rows
ydim, xdim = len(src_srcs), len(obs_srcs)
print('obs count: ', xdim, 'src count: ', ydim)
x = np.random.randint(10, size=ydim)
r = 1

p = 3 #int(np.ceil(xdim/2))
#
#Sx = random.sample(range(xdim), p)
#Sy = random.sample(range(ydim), p)
#
#C = np.transpose(np.array([B[:,col] for col in Sy]))
#R = (np.array([B[row,:] for row in Sx]))
#
#Uc,sc,Vc = lg.svd(C)
#
#U1 = Uc[:,0:r]
#U2 = C = (np.array([U[row,:] for row in Sx]))
#
#V = np.dot(R, lg.inv(U2))

U,s,V = lg.svd(B, full_matrices=0)
V = np.dot(np.diag(s), V)

Ur, Vr = U[:,0:r], V[0:r,:]
B2 = np.dot(Ur, Vr)

prod = np.dot(B,x)
prod2 = np.dot(B2, x)

error = (lg.norm(prod2) - lg.norm(prod))/ lg.norm(prod)
print(error)


