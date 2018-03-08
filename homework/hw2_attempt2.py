import time
import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import copy

row,col = 5,6 # row x row blocks of col x col toeplitz matrices
N = row*col

def idx2coord(i, c):
    row = np.floor(i/c)
    col = i%c
    return(row, col)
    
def fft_solve(a, x):
    aft, xft = np.fft.fft(a), np.fft.fft(x)
    return(np.fft.ifft(aft*xft))

def plot(g_mat):
    plt.imshow(g_mat, cmap='hot', interpolation='nearest')
    
def compute_g1(row, col):
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

def compute_blocks(row, col):
    blocks = []
    for y2 in range(row):
        block = np.array([np.zeros(col) for i in range(col)])
        for x1 in range(col):
            for x2 in range(col):
                if((x1 == x2) & (y2==0)):
                    block[x1, x2] = 1
                    continue;
                block[x1, x2] = 1 / np.sqrt((x2-x1)**2 + (y2)**2)
        blocks.append(block)
    return(blocks)

def compute_rowcol(row, col):
    rows = []
    cols = []
    for y2 in range(row):
        b_row = np.zeros(col)
        b_col = np.zeros(col)
        for x2 in range(col):
            if((x2 == 0) & (y2==0)):
                b_row[x2] = 1
                b_col[x2] = 0
                continue;
            b_row[x2] = 1 / np.sqrt((x2)**2 + (y2)**2)
            b_col[x2] = 1 / np.sqrt((-x2)**2 + (-y2)**2)
        rows.append(b_row)
        cols.append(b_col)
    return(rows, cols)

def create_circ_list(blocks):
    add_indices = np.roll(np.flip(np.array(range(len(blocks))), 0), 1)
    blocks_circ = copy.copy(blocks)
    for i in add_indices:
        blocks_circ.append(blocks[i])
    return(blocks_circ)

def toeplitz2circulant_b(tp_block):
    tp2_row = np.hstack((0, np.flip(tp_block[1:tp_block.shape[1],0], 0)))
    tp2_col = np.hstack((0, np.flip(tp_block[0,1:tp_block.shape[1]], 0)))
    tp2_block = lg.toeplitz(tp2_row, tp2_col)
    h0 = np.hstack((tp_block, tp2_block))
    h1 = np.hstack((tp2_block, tp_block))
    return(np.vstack((h0,h1)))

def toeplitz2circulant_r(row, col):
#    row is row of toeplitz mat
#    col is col of toeplitz mat
    work_col = np.roll(np.flip(col,0),1)
    work_col[0] = 0
    return(np.hstack((row,work_col)))

rows, cols = compute_rowcol(row, col)
blocks = compute_blocks(row, col)       
g_mat = compute_g1(row, col)

x = np.random.rand(N)
b_test = np.inner(g_mat,x)

# compute circulant matrices of tpz blocks
circ_blocks = [toeplitz2circulant_b(block) for block in blocks]
circ_rows = [toeplitz2circulant_r(rows[i], cols[i]) for i in range(row)]
circ_row = np.hstack(tuple(circ_rows))


# Create top row of circulant block circulant matrix
work_rows = [np.zeros(2*col) for i in range(row)]
for i in range(1,row):
    work_rows[row-i] = circ_rows[i]
block_circ_rows = circ_rows + work_rows
block_circ_row = np.hstack(tuple(block_circ_rows))

# Pad x vector
xp = np.zeros(4*N)
for i in range(row):
    xp[2*i*col:(2*i+1)*col] = x[i*col:(i+1)*col]
    
#### FAST MATVEC ####
# Diagnolize block_circ_row
pre_diag = np.transpose(np.reshape(block_circ_row,(2*row,2*col)))
diag = np.hstack(tuple((np.transpose(np.fft.fft2(pre_diag)))))

pre_xft = np.transpose(np.reshape(xp,(2*row,2*col)))
xft = np.hstack(tuple(np.transpose(np.fft.fft2(pre_xft))))

pre_b_fast = np.transpose(np.reshape(diag*xft,(2*row,2*col)))
b_fast_long = np.hstack(tuple(np.transpose(np.fft.ifft2(pre_b_fast))))
#### ####

#Reconstruct b
b_fast = np.zeros(N)
for i in range(row):
    b_fast[i*col:(i+1)*col] = b_fast_long[2*i*col:(2*i+1)*col]

circ_row = np.array([])
for block in circ_blocks:
    circ_row = np.hstack((circ_row, block[0,:]))
    



























#blocks_array = g_mat[0:col,:]
#copy = copy.copy(blocks_array)
#circ_array = np.array([np.zeros(2*N) for i in range(col)])
#circ_array[:,0:N] = blocks_array
#circ_array[:,N:2*N] = np.roll(np.flip(copy, 1), col)
#
#circ = np.array([np.zeros(2*N) for i in range(2*N)])
#
##plot_g(circ_array)
#
#x = np.random.rand(N)
#x_paddedft = np.hstack((np.fft.fft(x), np.zeros(N)))
#b_paddedft = np.zeros(2*N, dtype=np.complex)
#
#circ_arrayft = np.fft.fft2(circ_array)
#for block in range(row):
#    blockft = circ_arrayft[:,block * col: (block + 1) * col]
#    xft_chunk = x_paddedft[block * col: (block + 1) * col]
#    b_paddedft[block * col: (block + 1) * col] =  np.inner(blockft,xft_chunk)
#    
#b_padded = np.fft.ifft(b_paddedft)
#b_test = np.inner(g_mat, x)






