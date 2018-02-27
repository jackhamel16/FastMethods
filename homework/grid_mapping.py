import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import random

class source:
    def __init__(self, x, y, weight, N):
        self.x = x
        self.y = y
        self.weight = weight
        self.grid = np.array([0,0])
        self.pots = np.zeros(N)
        
def HeatPlot(mat):
    plt.imshow(mat, cmap='hot', interpolation='nearest')
        
# set up grid
Dx, Dy = 5,5
x_boxes, y_boxes = 5,5
step = Dx/x_boxes
grid = np.array([np.zeros(y_boxes+1) for i in range(x_boxes+1)])
Mrx, Mry = x_boxes, y_boxes
Msp = 1
tau = 1

N = 100
src_list = [source(1.5,1.5,1,1)]
#src_list = []
#for i in range(N):
#    src_list.append(source(Dx * np.random.random(), Dy * np.random.random(),\
#                    np.random.random(), N))
# Map to nearest bottom left corner
for src in src_list:
    src.grid = (int(np.floor(src.x/step)), int(np.floor(src.y/step)))


src = src_list[0]
order = 2
u_count = (order+1)**2# num of expansion points in Ca

l1, l2 = int(np.floor(order/2)), int(np.ceil(order/2)) # -l1, l2 range of expansion from grid pnt
weights = np.array([np.zeros(4) for i in range(4)])

def compute_Ca_shifts(u_count, l1, l2):
    # computes the shifts from the src's grid pnt given order of expansion
    count = 0
    shifts = np.array([np.zeros(2) for i in range(u_count)])
    for i in range(-l1,l2+1):
        for j in range(-l1,l2+1):
            shifts[count] = np.array([i,j])
            count += 1
    return(shifts)

Ca = np.array([src.grid for i in range(u_count)]) + \
         compute_Ca_shifts(u_count,l1,l2)
def compute_M(u_count, order):
    # Computes all possible cominations of m0, m1 given the order
    count = 0
    M = np.array([np.zeros(2) for i in range(u_count)])
    for i in range(order+1):
        for j in range(order+1):
            M[count] = np.array([i,j])
            count+= 1
    return(M)

M = compute_M(u_count, order)

W = np.array([np.zeros(u_count) for i in range(u_count)])
for ui,u in enumerate(Ca):
    ux = u-src.grid
    print(ux)
    for mi,m in enumerate(M):
        W[mi,ui] = np.prod(ux**m)

#Create Q vector
Q = np.zeros(u_count)
Q_terms = np.array([src.x,src.y]) - src.grid
for i in range(u_count):
    Q[i] = np.prod(Q_terms**M[i])
    
lam = np.inner(lg.inv(W),Q)

for i,u in enumerate(Ca):
    weights[int(u[0]),int(u[1])] = lam[i]
    
plt.scatter(Ca[:,0],Ca[:,1], c=lam)
plt.scatter(src.x,src.y, c=src.weight)
plt.colorbar()



