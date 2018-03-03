import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import random
import scipy.sparse as sparse

class source:
    def __init__(self, x, y, weight, N):
        self.x = x
        self.y = y
        self.weight = weight
        self.grid = np.array([0,0])
        self.near = []
        
def HeatPlot(mat):
    plt.imshow(mat, cmap='hot', interpolation='nearest')
    
def compute_Ca_shifts(u_count, l1, l2):
    # computes the shifts from the src's grid pnt given order of expansion
    # IMPORTANT: The values are relative to the current grid point
    count = 0
    shifts = np.array([np.zeros(2) for i in range(u_count)])
    for i in range(-l1,l2+1):
        for j in range(-l1,l2+1):
            shifts[count] = np.array([i,j])
            count += 1
    return(shifts)

def compute_M(u_count, order):
    # Computes all possible cominations of m0, m1 given the order
    count = 0
    M = np.array([np.zeros(2) for i in range(u_count)])
    for i in range(order+1):
        for j in range(order+1):
            M[count] = np.array([i,j])
            count+= 1
    return(M)

def compute_W(u_count, Ca, M):
    # computes all possible combinations of m0, m1
    W = np.array([np.zeros(u_count) for i in range(u_count)])
    for ui,ux in enumerate(Ca_shifts):
        for mi,m in enumerate(M):
            W[mi,ui] = np.prod(ux**m)
    return(W)

def compute_Q(u_count, M, src):
    Q = np.zeros(u_count)
    Q_terms = np.array([src.x,src.y]) - src.grid
    for i in range(u_count):
        Q[i] = src.weight*np.prod(Q_terms**M[i])
    return(Q)

def idx2coord(i, ctot):
    row = np.floor(i/ctot)
    col = i%ctot
    return(row, col)

def coord2idx(r, c, ctot):
    return(int(c + r * ctot))

# set up grid
Dx, Dy = 6,6
x_pnts, y_pnts = Dx+1, Dx+1
pnts = x_pnts * y_pnts
step = Dx/(x_pnts-1)

N = 2
src_list = [source(3,2.1,1,2), source(4.5,4.5,1,2)]
#src_list = []
#for i in range(N):
#    src_list.append(source(Dx * np.random.random(), Dy * np.random.random(),\
#                    np.random.random(), N))

# Map to nearest bottom left corner
for src in src_list:
    src.grid = (int(np.floor(src.x/step)), int(np.floor(src.y/step)))

order = 1
u_count = (order+1)**2# num of expansion points in Ca

l1, l2 = int(np.floor(order/2)), int(np.ceil(order/2)) # -l1, l2 range of expansion from grid pnt

Ca_shifts = compute_Ca_shifts(u_count, l1, l2)
M = compute_M(u_count, order)
W = compute_W(u_count, Ca_shifts, M)

src_maps = [[] for i in range(pnts)] # idx 1 is for grid pnt, idx 2 is list
                                     # of srcs based on their idx in src_list
#bnd_lo, bnd_hi = -l1-1, l2+1 # Bounds of near field region                  
#for i,src in enumerate(src_list):
#    src.grid = (int(np.floor(src.x/step)), int(np.floor(src.y/step)))
#    src_maps[coord2idx(src.grid[0], src.grid[1], x_pnts)].append(i)
#    # find and store near field locations for each src
#    for x_shift in range(bnd_lo, bnd_hi+1):
#        for y_shift in range(bnd_lo,bnd_hi+1):
#            nf_pnt = src.grid[0]+x_shift, src.grid[1]+y_shift
#            if (nf_pnt[0] > -1) & (nf_pnt[1] > -1) & \
#              (nf_pnt[0] <= x_pnts) & (nf_pnt[1] <= y_pnts):
#                src.near.append(coord2idx(x_shift+src.grid[0], 
#                                      y_shift+src.grid[1], x_pnts))

bnd_lo, bnd_hi = -l1-3, l2+2 # set bounds on where pnts are far field
#for src in src_list:
#    for i,src2 in enumerate(src_list):
#        delta_x, delta_y = src.grid[0]-src2.grid[0], src.grid[1]-src2.grid[1]
#        print(delta_x, delta_y)
#        if (bnd_lo < delta_x < bnd_hi) & (bnd_lo < delta_y < bnd_hi):
#            print(src.grid, src2.grid)
#            src.near.append(i)

lam_mat = sparse.lil_matrix((N,pnts))
for i,src in enumerate(src_list):
    src.grid = (int(np.floor(src.x/step)), int(np.floor(src.y/step)))
    
    for j,src2 in enumerate(src_list): # find srcs in near field of src
        delta_x, delta_y = src.grid[0]-src2.grid[0], src.grid[1]-src2.grid[1]
        if (bnd_lo < delta_x < bnd_hi) & (bnd_lo < delta_y < bnd_hi):
            src.near.append(j)
    
    Q = compute_Q(u_count, M, src)
    lam = np.inner(lg.inv(W), Q)
    for j,u in enumerate(Ca_shifts):
        idx = coord2idx(int(u[0]+src.grid[0]),int(u[1]+src.grid[1]),x_pnts)
        lam_mat[i,idx] = lam[j] 

lam_mat = lam_mat.asformat("csr") #better suited for mutliplication

#lam_G = lam_mat*G
#A_fn = lam_G*lam_mat.transpose()

#lam_mat = np.array([np.zeros(Dx*Dx) for i in range(N)])
#Q_mat = np.array([np.zeros(u_count) for i in range(N)])
#for n,src in enumerate(src_list):
#    Q_mat[n,:] = compute_Q(u_count, M, src)
#lam_mat = np.inner(lg.inv(W),Q_mat)

#plt.scatter(Ca[:,0],Ca[:,1], c=lam)
#plt.scatter(src.x,src.y, c=src.weight)
#plt.colorbar()


# NON SPARSE LAMBDA COMPUTATION
#lam_mat = np.array([np.zeros(pnts) for i in range(N)])
#for i,src in enumerate(src_list):    
#    uniform_grid = np.array([np.zeros(x_pnts) for i in range(y_pnts)])
#    Q = compute_Q(u_count, M, src)
#    lam = np.inner(lg.inv(W), Q)
#    print(lam)
#    for j,u in enumerate(Ca_shifts):
#        print(int(u[0]+src.grid[0]),int(u[1]+src.grid[1]))
#        uniform_grid[int(u[0]+src.grid[0]),int(u[1]+src.grid[1])] = lam[j]
#    lam_mat[i,:] = np.hstack(uniform_grid)
#lam_G = np.matmul(lam_mat, G)
#A = np.matmul(lam_G, np.transpose(lam_mat))
