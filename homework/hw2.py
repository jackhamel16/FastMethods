import time
import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import copy

row,col = 2,2 # row x row blocks of col x col toeplitz matrices
N = row*col

#def idx2coord(i,N):
#    row = i%np.sqrt(N)
#    col = np.floor(i/np.sqrt(N))
#    return(int(col), int(row))

def idx2coord(i, c):
    row = np.floor(i/c)
    col = i%c
    return(row, col)
    
def fft_solve(a, x):
    aft, xft = np.fft.fft(a), np.fft.fft(x)
    return(np.fft.ifft(aft*xft))
    
g_mat = np.array([np.zeros(N) for i in range(N)])

for i in range(N):
    x_src, y_src = idx2coord(i, col)
    for j in range(N):
        if(i==j):
            g_mat[i,j] = 1
            continue
        x_obs, y_obs = idx2coord(j, col)
        g_mat[i,j] = 1/(np.sqrt((x_obs-x_src)**2+(y_obs-y_src)**2))

blk_dim, tpz_dim = row, col
x = np.random.rand(N)

t0vec = g_mat[0,0:col]
t0circvec = np.hstack((t0vec, np.roll(np.flip(t0vec, 0), 1)))
t1vec = g_mat[0,col:2*col]
t1circvec = np.hstack((t1vec, np.roll(np.flip(t1vec, 0), 1)))

x0 = np.hstack((x[0:col], np.zeros(col)))
x1 = np.hstack((np.zeros(col), x[col:N]))
t0circ = lg.circulant(t0circvec)
t1circ = lg.circulant(t1circvec)
 
t0x0 = np.inner(t0circ, x0)
t0x1 = np.inner(t0circ, x1)
t1x0 = np.inner(t1circ, x0)
t1x1 = np.inner(t1circ, x1)

b0test = np.hstack((t0x0[0:col], t0x1[col-1:-1]))
b1test = np.hstack((t1x1[col-11:-1], t1x0[0:col]))

btest = b0test + b1test
b = np.inner(g_mat, x)

#t0 = g_mat[0:col,0:col]
#t0_vec = np.hstack((t0[0,0:col], t0[-1,0:col]))
#t0_circ = lg.circulant (t0_vec)
#t0_0 = fft_solve(t0_vec, x[0:2*col])[0:col]
#
#t1 = g_mat[col:2*col]
#t1_vec = np.hstack((t1[0,0:col], t1[-1,0:col]))
#t1_0 = fft_solve(t1_vec, x[0:2*col])[col:2*col]
#
#x = np.ones(N)
b = np.inner(g_mat,x)


plt.imshow(g_mat, cmap='hot', interpolation='nearest')