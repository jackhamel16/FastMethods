import numpy as np
import numpy.linalg as lg
import time
import interaction
import source
import utilities as utils
import scipy.special as funcs
    
N = 10

row_tot, col_tot = 4, 4
step = 1
Dx, Dy = row_tot * step, col_tot * step
box_tot = row_tot * col_tot

box_list = [[] for i in range(box_tot)]
src_list = []
for i in range(N):
    src_list.append(source.source(Dx * np.random.random(), Dy * \
                                  np.random.random(), np.random.random()))
    # Map src to nearest lower left grid pnt
    src_list[i].grid = np.array([int(np.floor(src_list[i].x/step)), \
                                 int(np.floor(src_list[i].y/step))])
    src_list[i].idx = utils.coord2idx(src_list[i].grid[0], \
                                      src_list[i].grid[1], col_tot)
    box_list[src_list[i].idx].append(i)
    
interactions = interaction.interaction(box_tot, col_tot, row_tot, \
                                       src_list, box_list)
    
interactions.fill_lists()
test = interactions.list


#Calculate Multipoles
k = 2 * np.pi / 0.1
alpha_list = [i for i in np.arange(0, 2*np.pi+1, 0.5*np.pi)]
multipoles = [[] for i in range(box_tot)]
for box_idx in range(box_tot):
    for i, alpha in enumerate(alpha_list):
        val = 0
        for src_idx in box_list[box_idx]:
            src = src_list[src_idx]
            val += np.exp(-np.complex(0,1) * k * src.rho * \
                          np.cos(alpha - src.theta)) * src.weight
        multipoles[box_idx].append(val)
        
#M2M
M2L_list = [[] for i in range(box_tot)]
P = 5
for obs_box_idx in range(box_tot):
    obs_x, obs_y = utils.idx2coord(obs_box_idx, col_tot)
    for src_box_idx in interactions.list[obs_box_idx]:
        vals = []
        for alpha in alpha_list:
            src_x, src_y = utils.idx2coord(src_box_idx, col_tot)
            x, y = obs_x - src_x, obs_y - src_y
            sep_rho, sep_theta = utils.cart2cyl(x, y)
            
            val = 0
            for p in np.arange(-P, P+1, 1):
                val += funcs.hankel1(p, k*sep_rho) * np.exp(np.complex(0,1) * \
                                     p * (sep_theta - alpha - np.pi/2))
            vals.append(val)
        M2L_list[obs_box_idx].append(vals)
            
# interactions
for obs_box_idx in range(box_tot):
    obs_mlist = multipoles[obs_box_idx]
    for j,src_box_idx in enumerate(interactions.list[obs_box_idx]):
        src_mlist = multipoles[obs_box_idx]
        M2L = M2L_list[obs_box_idx][j]
        











