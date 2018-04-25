import numpy as np
import numpy.linalg as lg
import time
import interaction
import source
import utilities as utils
import scipy.special as funcs
    
N = 500
step = 1
modifier = 1

row_tot, col_tot = int(np.round(np.sqrt(N/10)/step)), \
                   int(np.round(np.sqrt(N/10)/step))
#row_tot, col_tot = 8,8
Dx, Dy = row_tot * step, col_tot * step
box_tot = row_tot * col_tot

k = 2 * np.pi / (step * row_tot)
P = int(np.ceil(k * np.sqrt(2) * step)) * modifier# proportional to box diameter * k
Q = 2 * P + 1
# source creation
box_list = [[] for i in range(box_tot)]
src_list = []
#src_list = [source.source(0.3, 0.5, 1),source.source(3.4,3.5,1)]#source.source(3.5, 3.3, 1)]
for i in range(N):
    src_list.append(source.source(Dx * np.random.random(), Dy * \
                                  np.random.random(), np.random.random()))
    # Map src to nearest lower left grid pnt
    src_list[i].grid = np.array([int(np.floor(src_list[i].x/step)), \
                                 int(np.floor(src_list[i].y/step))])
    src_list[i].idx = utils.coord2idx(src_list[i].grid[0], \
                                      src_list[i].grid[1], col_tot)
    # compute c2m vector in cyl
    src_list[i].rho, src_list[i].theta = utils.cart2cyl((src_list[i].grid[0] +\
      0.5)*step - src_list[i].x, (src_list[i].grid[1] + 0.5)*step - src_list[i].y)
    # contains source idxs in each box
    box_list[src_list[i].idx].append(i)
    
interactions = interaction.interaction(box_tot, col_tot, row_tot, \
                                       src_list, box_list)
interactions.fill_lists()
test = interactions.list
  
#Calculate Multipoles
alpha_list = np.array([i for i in np.arange(0, 2*np.pi, 2*np.pi/Q)])
C2M_list = [[] for i in range(box_tot)]
for box_idx in range(box_tot):
    for i, alpha in enumerate(alpha_list):
        val = 0
        for src_idx in box_list[box_idx]:
            src = src_list[src_idx]
            val += np.exp(np.complex(0,1) * k * src.rho * \
                          np.cos(alpha - src.theta)) * src.weight
        C2M_list[box_idx].append(val)
        
#M2L
M2L_list = [[0 for i in range(box_tot)] for i in range(box_tot)]
for obs_box_idx in range(box_tot):
    obs_x, obs_y = np.array(utils.idx2coord(obs_box_idx, col_tot))*step
    for src_box_idx in interactions.list[obs_box_idx]:
        vals = []
        for alpha in alpha_list:
            src_x, src_y = np.array(utils.idx2coord(src_box_idx, col_tot))*step
            x, y = obs_x - src_x, obs_y - src_y
            sep_rho, sep_theta = utils.cart2cyl(x, y)
            val = 0
            for p in np.arange(-P, P+1, 1):
                val += funcs.hankel1(p, k*sep_rho) * np.exp(-np.complex(0,1) *\
                                     p * (sep_theta - alpha - np.pi/2))
            vals.append(val)
        M2L_list[obs_box_idx][src_box_idx] += np.array(vals)

#L2O
L2O_list = [[] for i in range(N)]
for i, src in enumerate(src_list):
    for alpha in alpha_list:
        L2O_list[i].append(np.exp(np.complex(0,1) * k * src.rho * \
                        np.cos(alpha - (src.theta + np.pi))))
            
# interactions
pot = np.array([np.complex(0,0) for i in range(N)])
for obs_box_idx in range(box_tot):
    C2L_list = []
    for i, src_box_idx in enumerate(interactions.list[obs_box_idx]):
        # translates from sources to local multipole
        C2L_list.append(M2L_list[obs_box_idx][src_box_idx] * \
                        C2M_list[src_box_idx])
    for i, obs_idx in enumerate(box_list[obs_box_idx]):
        C2O_list = [L2O_list[obs_idx] * C2L for C2L in C2L_list]
        pot[obs_idx] = np.sum(C2O_list) / Q
    # near interactions
    near_pot = interactions.compute_near(obs_box_idx, k)
    for i, obs_idx in enumerate(box_list[obs_box_idx]):
        pot[obs_idx] += near_pot[i] 
        
# TESTING
src_idxs = [i for i in range(N)]
G = interactions.build_G(src_idxs, src_idxs, k)
weights = np.array([src.weight for src in src_list])
test_pot = np.dot(G, weights)

error = (lg.norm(pot) - lg.norm(test_pot)) / lg.norm(test_pot)
print(error)
        











