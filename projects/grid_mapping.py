import numpy as np
import scipy.linalg as lg
import scipy.sparse as sparse

class source:
    def __init__(self, x, y, weight, N):
        self.x = x
        self.y = y
        self.weight = weight
        self.grid = np.array([0,0])
        self.near = []

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

def compute_directly(src_list, N):
    A_direct = np.array([np.zeros(N) for i in range(N)])
    for i,src1 in enumerate(src_list):
        for j,src2 in enumerate(src_list):
            if (src1.x != src2.x) or (src1.y != src2.y):
                A_direct[i,j] = src1.weight * src2.weight / \
                                np.sqrt((src1.x - src2.x)**2 + \
                                        (src1.y - src2.y)**2)
    return(A_direct)

def compute_M(u_count, order):
    # Computes all possible cominations of m0, m1 given the order
    count = 0
    M = np.array([np.zeros(2) for i in range(u_count)])
    for i in range(order+1):
        for j in range(order+1):
            M[count] = np.array([i,j])
            count+= 1
    return(M)

def compute_near_fields(src_list, lam_mat, lam_G, N):
    A_n = sparse.lil_matrix((N,N)) # Direct near fields
    A_nf = sparse.lil_matrix((N,N)) # Near fields with grid mapping
    
    for i, src in enumerate(src_list):
        for j, near_src in enumerate(src.near):
            A_nf[i,near_src] = lam_G[i,:] * lam_mat[near_src,:].transpose()
            # Direct near field computation:
            if (src.x != src_list[near_src].x) or \
               (src.y != src_list[near_src].y):
                A_n[i,near_src] = src.weight * src_list[near_src].weight / \
                                  np.sqrt((src.x - src_list[near_src].x)**2 + \
                                          (src.y - src_list[near_src].y)**2)
    return(A_n.asformat("csr"), A_nf.asformat("csr"))

def compute_W(u_count, Ca_shifts, M):
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

def find_near_srcs(src, src_list, lo, hi):
    for i,src2 in enumerate(src_list): # find srcs in near field of src
        delta_x, delta_y = src.grid[0]-src2.grid[0], src.grid[1]-src2.grid[1]
        if (lo < delta_x < hi) & (lo < delta_y < hi):
            src.near.append(i)


# CODE TO FIND NEAR FIELD PINTS OF A SRC
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
