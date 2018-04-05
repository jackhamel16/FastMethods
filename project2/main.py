import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import time
import interaction
import source
import tree
import utilities as utils

#level_cnt = 4 # Count levels starting from root = 0 
#grid_step = 1
#N = 1000
#eps = 1e-6

def run(level_cnt, grid_step, N, eps):
    grid_dim = 2**(level_cnt-1) # Should remain power of two for easy life
    src_list = []
    for i in range(N):
        src_list.append(source.source(grid_dim * np.random.random(),grid_dim *\
                                  np.random.random(), np.random.random()))
        # Map src to nearest lower left grid pnt
        src_list[i].grid = (int(np.floor(src_list[i].x/grid_step)), \
                            int(np.floor(src_list[i].y/grid_step)))
    
    print("Building Tree...")    
    my_tree = tree.tree(src_list, level_cnt)
    my_tree.build()
    
    print("Filling Interaction Lists...")
    interactions = interaction.interaction(level_cnt, my_tree)
    interactions.fill_list()
    
    leaf_start = 2**(2*(level_cnt-1))
    leaf_end = 2*leaf_start
    
    for obs_idx in range(leaf_start, leaf_end):
        for src_idx in range(leaf_start, leaf_end):
            G = interactions.build_G(my_tree.tree[obs_idx], \
                                     my_tree.tree[src_idx])
            if (my_tree.tree[src_idx] == []) or (my_tree.tree[obs_idx] == []):
                U, V = np.array([]), np.array([])
            else:
                U,V = utils.uv_decompose(G, eps)
            interactions.uv_list[obs_idx][src_idx] = (U,V)
    
    print('Computing UV Decompositions...')
    for lvl in range(level_cnt-2, 1, -1):
        lb = 2**(2*lvl)
        ub = 2*lb
        for obs_idx in range(lb,ub):
            for src_idx in interactions.list[obs_idx]:
    #        for src_idx in range(lb,ub):
                n = my_tree.get_children(obs_idx,lvl) #rows of merging
                m = my_tree.get_children(src_idx,lvl) #cols of merging
                uv = [[0,0],[0,0]] # index as [row][col]
                for i in range(2):
                    for j in range(2):
                        U1, V1 = interactions.uv_list[n[2*i]][m[2*j]]
                        U2, V2 = interactions.uv_list[n[2*i+1]][m[2*j]]
                        U3, V3 = interactions.uv_list[n[2*i]][m[2*j+1]]
                        U4, V4 = interactions.uv_list[n[2*i+1]][m[2*j+1]]
                        
                        U12,V12 = utils.merge(U1, V1, U2, V2, eps)
                        U34,V34 = utils.merge(U3, V3, U4, V4, eps)
                        # Horizontal merge
                        uv[i][j] = utils.merge(U12, V12, U34, V34, eps, 1)
                
                Um1,Vm1 = utils.merge(uv[0][0][0], uv[0][0][1],\
                                      uv[1][0][0], uv[1][0][1], eps)
                Um2,Vm2 = utils.merge(uv[0][1][0], uv[0][1][1], \
                                      uv[1][1][0], uv[1][1][1], eps)
                
                U,V = utils.merge(Um1, Vm1, Um2, Vm2, eps, 1)
                interactions.uv_list[obs_idx][src_idx] = (U, V)  
    
    fast_time = 0    
    print("Computing Fast Interactions...")
    for obs_box_idx in range(len(interactions.list)):
        obs_srcs = my_tree.tree[obs_box_idx]
        obs_pot = np.zeros(len(obs_srcs))
        for src_box_idx in interactions.list[obs_box_idx]:
            src_srcs = my_tree.tree[src_box_idx]
            src_vec = np.array([src_list[idx].weight for idx in src_srcs])
            U, V = interactions.uv_list[obs_box_idx][src_box_idx]
            if np.size(U) != 0:
                s = time.clock() 
                obs_pot += np.dot(U, np.dot(V, src_vec))
                fast_time += time.clock() - s
        #near field interacitons
        obs_pot += interactions.compute_near(obs_box_idx)
        for i, obs in enumerate(obs_srcs):
            s = time.clock()
            interactions.potentials[obs] += obs_pot[i]
            fast_time += time.clock() - s
    
    #Direct Computation
    print("Computing Direct Interactions...")
    idxs = [i for i in range(N)]
    G = interactions.build_G(idxs, idxs)
    src_vec = np.array([src.weight for src in src_list])
    s = time.clock()
    direct_potentials = np.dot(G, src_vec)
    slow_time = time.clock() - s
    #
    error = (lg.norm(interactions.potentials) - lg.norm(direct_potentials))\
            / lg.norm(direct_potentials)
            
    print('Error: ', error)
    print('Fast Time: ', fast_time)
    print('Slow Time: ', slow_time)
    
    return(fast_time, slow_time, error)
    
## old testing code but saving it just incase ### 

#lvl = 2
#obs_idx = 16
#src_idx = 25
#n = my_tree.get_children(obs_idx,lvl) #rows of merging
#m = my_tree.get_children(src_idx,lvl) #cols of merging
#rank = 1
#uv = [[0,0],[0,0]] # index as [row][col]
#for i in range(2):
#    for j in range(2):
#        print(i,j)
#        U1, V1 = interactions.uv_list[n[2*i]][m[2*j]]
#        U2, V2 = interactions.uv_list[n[2*i+1]][m[2*j]]
#        U3, V3 = interactions.uv_list[n[2*i]][m[2*j+1]]
#        U4, V4 = interactions.uv_list[n[2*i+1]][m[2*j+1]]
#        
#        U12,V12 = utils.merge(U1, V1, U2, V2, eps)
#        U34,V34 = utils.merge(U3, V3, U4, V4, eps)
#        # Horizontal merge
#        uv[i][j] = utils.merge(U12, V12, U34, V34, eps, 1)
#
#Um1,Vm1 = utils.merge(uv[0][0][0], uv[0][0][1],\
#                      uv[1][0][0], uv[1][0][1], eps)
#Um2,Vm2 = utils.merge(uv[0][1][0], uv[0][1][1], \
#                      uv[1][1][0], uv[1][1][1], eps)
#
#U,V = utils.merge(Um1, Vm1, Um2, Vm2, eps, 1)