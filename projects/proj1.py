import time
import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
    
def block_toeplitz2block_circulant(circ_rows, row, col):
    work_rows = [np.zeros(2*col) for i in range(row)]
    for i in range(1,row):
        work_rows[row-i] = circ_rows[i]
    block_circ_rows = circ_rows + work_rows
    return(np.hstack(tuple(block_circ_rows)))
    
def compute_g1(row, col):
    # Directly computes entire G matrix. Useful for testing, not practical
    N = row*col
    g_mat = np.array([np.zeros(N) for i in range(N)])
    for i in range(N):
        x_src, y_src = idx2coord(i, col)
        for j in range(N):
            if(i==j):
                g_mat[i,j] = 1
                continue
            x_obs, y_obs = idx2coord(j, col)
            g_mat[i,j] = 1/(np.sqrt((x_obs-x_src)**2+(y_obs-y_src)**2))
    return(g_mat)

def compute_rowcol(row, col):
    # Computes first row and col of G mat.=, which fully describe system
    rows, cols = [], []
    for y2 in range(row):
        b_row, b_col = np.zeros(col), np.zeros(col)
        for x2 in range(col):
            if((x2 == 0) & (y2==0)):
                b_row[x2], b_col[x2] = 1, 0
                continue;
            b_row[x2] = 1 / np.sqrt((x2)**2 + (y2)**2)
            b_col[x2] = 1 / np.sqrt((-x2)**2 + (-y2)**2)
        rows.append(b_row)
        cols.append(b_col)
    return(rows, cols)

def fft_matvec(block_circ_row, xp, row, col):
    #Computes the fast matvec of a circulant block circulant mat given its
    #top row with x (which is padded in this application)
    pre_diag = reshape(block_circ_row,row, col)
    diag = reshape_inv(np.fft.fft2(pre_diag))
    pre_xft = reshape(xp,row,col)
    xft = reshape_inv(np.fft.fft2(pre_xft))
    pre_b_fast = reshape(diag*xft, row, col)
    b_fast_long = reshape_inv(np.fft.ifft2(pre_b_fast))
    
    b_fast = np.zeros(N)
    for i in range(row):  # removing solutions due to padding of x
        b_fast[i*col:(i+1)*col] = np.real(b_fast_long[2*i*col:(2*i+1)*col])
    return(b_fast)

def idx2coord(i, c):
    row = np.floor(i/c)
    col = i%c
    return(row, col)

def plot(mat):
    # Quick way to visualize matrices
    plt.imshow(mat, cmap='hot', interpolation='nearest')

def reshape(vec, row, col):
    return(np.transpose(np.reshape(vec,(2*row,2*col))))

def reshape_inv(mat):
    return(np.hstack(tuple(np.transpose(mat))))

def toeplitz2circulant_row(row, col):
    # Converts top row of toeplitz mat to top row of circulant mat
    # row is row of toeplitz mat, col is col of toeplitz mat
    work_col = np.roll(np.flip(col,0),1)
    work_col[0] = 0
    return(np.hstack((row,work_col)))

row,col = 5,6 # row x row blocks of col x col toeplitz matrices
N = row*col
x = np.random.rand(N)

rows, cols = compute_rowcol(row, col) # of G matrix
circ_rows = [toeplitz2circulant_row(rows[i], cols[i]) for i in range(row)]
block_circ_row = block_toeplitz2block_circulant(circ_rows, row, col)

# Pad x vector
xp = np.zeros(4*N)
for i in range(row):
    xp[2*i*col:(2*i+1)*col] = x[i*col:(i+1)*col]
    
b_fast = fft_matvec(block_circ_row, xp, row, col)