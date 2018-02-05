import time
import numpy as np
import matplotlib.pyplot as plt
import copy

size = 3
N = size**2

def idx2coord(i,N):
    row = i%np.sqrt(N)
    col = np.floor(i/np.sqrt(N))
    return(int(col), int(row))
    
g_mat = np.array([np.zeros(size**2) for i in range(size**2)])
x_mat = np.array([np.zeros(size) for i in range(size)])
y_mat = copy.copy(x_mat)

N_vec = np.array([i for i in range(N)])
for i in range(N):
    x_src, y_src = idx2coord(i, N)
#    print(x_src,y_src)
    for j in N_vec:
#        if(i==2):
#            print(x_obs,y_obs)
        x_obs, y_obs = idx2coord(j, N)
#        if(i==j):
#            g_mat[i,j] = 1
#            continue
        g_mat[i,j] = (np.sqrt((x_obs-x_src)**2+(y_obs-y_src)**2))
#    N_vec = np.roll(N_vec,-1)
#    print(N_vec)

# FIGURE OUT HOW TO GO FROM TOEPLITZ -> CIRCULENT

g_vec = np.zeros(2*N)
g_vec[0:N] = g_mat[0,:]
g_vec[N:2*N] = g_mat[1,:]       

circ_mat = np.array([np.zeros(2*size**2) for i in range(2*size**2)])
for i in range(2*N):
    circ_mat[i,:] = g_vec
    g_vec = np.roll(g_vec,1)

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