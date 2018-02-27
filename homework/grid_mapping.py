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
        
# set up grid
Dx, Dy = 5,5
x_boxes, y_boxes = 5,5
step = Dx/x_boxes
grid = np.array([np.zeros(y_boxes+1) for i in range(x_boxes+1)])
Mrx, Mry = x_boxes, y_boxes
Msp = 1
tau = 1

N = 100
src_list = [source(1.1,1.6,1,1)]
#src_list = []
#for i in range(N):
#    src_list.append(source(Dx * np.random.random(), Dy * np.random.random(),\
#                    np.random.random(), N))
# Map to nearest bottom left corner
for src in src_list:
    src.grid = (int(np.floor(src.x/step)), int(np.floor(src.y/step)))


src = src_list[0]
order = 2
idxs = (order+1)**2    
W = np.array([np.zeros(idxs) for i in range(idxs)])
Ca = np.array([[1,1],[1,2],[2,1],[2,2]])
M_check = np.array([[0,0],[0,1],[1,0],[1,1]])

count = 0
M = np.array([np.zeros(2) for i in range(idxs)])
for i in range(order+1):
    for j in range(order+1):
        M[count] = np.array([i,j])
        count+= 1

#for ui,u in enumerate(Ca):
#    ux = u-src.grid
#    for mi,m in enumerate(M):
#        W[mi,ui] = np.prod(ux**m)
##Create Q vector
#Q = np.zeros(idxs)
#Q_terms = np.array([src.x,src.y]) - src.grid
#for i in range(4):
#    Q[i] = np.prod(Q_terms**M[i])
    
#lam = np.inner(lg.inv(W),Q)