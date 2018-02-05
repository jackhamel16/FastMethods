import time
import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import copy

row,col = 3,20
N = row*col

#def idx2coord(i,N):
#    row = i%np.sqrt(N)
#    col = np.floor(i/np.sqrt(N))
#    return(int(col), int(row))

def idx2coord(i,c):
    row = np.floor(i/c)
    col = i%c
    return(int(row), int(col))
    
g_mat = np.array([np.zeros(N) for i in range(N)])

N_vec = np.array([i for i in range(N)])
for i in range(N):
    x_src, y_src = idx2coord(i, col)
    for j in range(N):
        x_obs, y_obs = idx2coord(j, col)
        g_mat[i,j] = (np.sqrt((x_obs-x_src)**2+(y_obs-y_src)**2))
        

g_mat2 = np.array([np.zeros(col**2) for i in range(row**2)])

blocks = []
for i in range(row):
    dist = np.zeros(col)
    for x in range(col):
        dist[x] = np.sqrt((x - 0)**2 + (0 - i)**2)
    tpz = lg.toeplitz(dist, dist)
    blocks.append(tpz)
    
#    for y in range(row):
#        dist[]


# FIGURE OUT HOW TO GO FROM TOEPLITZ -> CIRCULENT

#g_vec = np.zeros(2*N)
#g_vec[0:N] = g_mat[0,:]
#g_vec[N:2*N] = g_mat[1,:]       
#
#circ_mat = np.array([np.zeros(2*size**2) for i in range(2*size**2)])
#for i in range(2*N):
#    circ_mat[i,:] = g_vec
#    g_vec = np.roll(g_vec,1)

#circ_mat[0:N,0:N] = copy.copy(g_mat)
#circ_mat[0:N,N:2*N] = copy.copy(g_mat)
#circ_mat[N:2*N,0:N] = copy.copy(g_mat)
#circ_mat[N:2*N,N:2*N] = copy.copy(g_mat)



plt.imshow(g_mat, cmap='hot', interpolation='nearest')

#plt.imshow(circ_mat, cmap='hot', interpolation='nearest')
#for i in range(2*N):
#    for j in range(2*N):
#        print("{a:2.2f}".format(a=circ_mat[i,j]), end= "  ")
#    print("\n")