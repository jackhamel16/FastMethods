import time
import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import scipy.sparse as sparse

import grid_mapping as gm

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

def coord2idx(r, c, ctot):
    return(int(c + r * ctot))

def pad_x_vector(x, N, col):
    xp = np.zeros(4*N)
    for i in range(row):
        xp[2*i*col:(2*i+1)*col] = x[i*col:(i+1)*col]
    return(xp)

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

# SETUP #
N = 10
Dx, Dy = 10, 10
order = 1
rows, cols = Dx+1, Dx+1 # rowsXrows blocks of colsXcols toeplitz matrices
N_grid = rows * cols;
xstep, ystep = Dx/(rows-1), Dy/(rows-1)
u_count = (order+1)**2# num of expansion points in Ca

# -l1, l2 range of expansion from grid pnt
l1, l2 = int(np.floor(order/2)), int(np.ceil(order/2)) 
# set bounds on where pnts are not far field
bnd_lo, bnd_hi = -l1-3, l2+2

x = np.random.rand(N)

# Generate sources (srcs):
src_list = []
for i in range(N):
    src_list.append(gm.source(Dx * np.random.random(),Dy * np.random.random(),\
                    np.random.random(), N))
    # Map src to nearest lower left grid pnt
    src_list[i].grid = (int(np.floor(src_list[i].x/xstep)), \
                        int(np.floor(src_list[i].y/ystep)))

# Begin Mapping
Ca_shifts = gm.compute_Ca_shifts(u_count, l1, l2)
M = gm.compute_M(u_count, order)
W = gm.compute_W(u_count, Ca_shifts, M)

Winv = lg.inv(W)
lam_mat = sparse.lil_matrix((N,N_grid))
print("Computing Lambda and Finding Near Sources...")
for i,src in enumerate(src_list):
    gm.find_near_srcs(src, src_list, bnd_lo, bnd_hi)
    
    Q = gm.compute_Q(u_count, M, src)
    lam = np.inner(Winv, Q)
    for j,u in enumerate(Ca_shifts):
        idx = coord2idx(int(u[0]+src.grid[0]),int(u[1]+src.grid[1]),rows)
        lam_mat[i,idx] = lam[j] 

lam_mat = lam_mat.asformat("csr") #better suited for mutliplication
print("Compute G")
G = compute_g1(rows,cols)
print("Computing Lambda * G...")
lam_G = lam_mat*G
print("Computing Near-Far Fields...")    
A_near, An = gm.compute_near_fields(src_list, lam_mat, N)
A_fn = lam_G*lam_mat.transpose()

# Direct Computation
print("Computing Direct Interactions...")

A_direct = compute_directly(src_list, N)
A = A_near + A_fn - An

pot_aim = sum(A.transpose())
pot_direct = sum(A_direct.transpose())

error = (pot_aim - pot_direct) / pot_direct
avg_error = error.sum()/N
print(avg_error)

#xp = pad_x_vector(x,N,cols)
#
#G_rows, G_cols = compute_rowcol(row, col) # of G matrix
#circ_rows = [toeplitz2circulant_row(G_rows[i], G_cols[i]) for i in range(rows)]
#block_circ_row = block_toeplitz2block_circulant(circ_rows, rows, cols)
#
#    
#b_fast = fft_matvec(block_circ_row, xp, rows, cols)