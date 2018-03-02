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
x_pnts, y_pnts = Dx, Dx
pnts = x_pnts * y_pnts
step = Dx/(x_boxes+1)

N = 2
src_list = [source(3,0,1,2), source(4.5,4.5,1,2)]
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
    
lam_mat = sparse.lil_matrix((N,pnts))
for i,src in enumerate(src_list):
    src.grid = (int(np.floor(src.x/step)), int(np.floor(src.y/step)))    
    
    
    Q = compute_Q(u_count, M, src)
    lam = np.inner(lg.inv(W), Q)
    for j,u in enumerate(Ca_shifts):
        idx = coord2idx(int(u[0]+src.grid[0]),int(u[1]+src.grid[1]),x_pnts)
        lam_mat[i,idx] = lam[j] 

lam_mat = lam_mat.asformat("csr") #better suited for mutliplication

lam_G = lam_mat*G
A_fn = lam_G*lam_mat.transpose()

#lam_mat = np.array([np.zeros(Dx*Dx) for i in range(N)])
#Q_mat = np.array([np.zeros(u_count) for i in range(N)])
#for n,src in enumerate(src_list):
#    Q_mat[n,:] = compute_Q(u_count, M, src)
#lam_mat = np.inner(lg.inv(W),Q_mat)

#plt.scatter(Ca[:,0],Ca[:,1], c=lam)
#plt.scatter(src.x,src.y, c=src.weight)
#plt.colorbar()


# NON SPARSE LAMBDA COMPUTATION
#lam_mat = np.array([np.zeros(Dx*Dx) for i in range(N)])
#for i,src in enumerate(src_list):    
#    uniform_grid = np.array([np.zeros(Dx) for i in range(Dy)])
#    Q = compute_Q(u_count, M, src)
#    lam = np.inner(lg.inv(W), Q)
#    for j,u in enumerate(Ca_shifts):
#        uniform_grid[int(u[0]+src.grid[0]),int(u[1]+src.grid[1])] += lam[j]
#    lam_mat[i,:] = np.hstack(uniform_grid)
#lam_G = np.matmul(lam_mat, G)
#A = np.matmul(lam_G, np.transpose(lam_mat))
