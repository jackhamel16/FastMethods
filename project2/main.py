import numpy as np
import numpy.linalg as lg
import matplotlib.pyplot as plt
import interaction
import source
import tree

level_cnt = 5 # Count levels starting from root = 0 
grid_dim = 2**(level_cnt-1) # Should remain power of two for easy life
grid_step = 1
N = 500


src_list = []
#src_list = [source.source(0.1,0.1,1), source.source(0.4,0.4,1), \
#            source.source(3.5,3.5,1), source.source(2.1,2.1,1), \
#            source.source(3.2,3.2,1)]
for i in range(N):
    src_list.append(source.source(grid_dim * np.random.random(),grid_dim * \
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

#Computing Potentials
print("Computing Fast Interactions...")
for obs_box_idx in range(len(interactions.list)):
    obs_srcs = my_tree.tree[obs_box_idx]
    obs_pot = np.zeros(len(obs_srcs))
    #far field interactions
    obs_pot = interactions.compute_box_pot_slow(obs_box_idx)
    #near field interacitons
    obs_pot += interactions.compute_box_pot_slow(obs_box_idx, 1)
    for i, obs in enumerate(obs_srcs):
        interactions.potentials[obs] += obs_pot[i]

#Direct Computation
print("Computing Direct Interactions...")
idxs = [i for i in range(N)]
G = interactions.build_G(idxs, idxs)
src_vec = np.array([src.weight for src in src_list])
direct_potentials = np.dot(G, src_vec)

error = (lg.norm(interactions.potentials) - lg.norm(direct_potentials))\
        / lg.norm(direct_potentials)
        
print('Error: ', error)

error_vec = (interactions.potentials - direct_potentials) / direct_potentials

### OLD
#for obs_box_idx in range(len(interactions.list)):
#    obs_srcs = my_tree.tree[obs_box_idx]
#    obs_pot = np.zeros(len(obs_srcs))
#    
#    #far field interactions
#    for src_box_idx in interactions.list[obs_box_idx]:
#        src_srcs = my_tree.tree[src_box_idx]
#        src_vec = np.array([src_list[idx].weight for idx in src_srcs])
#        G = interactions.build_G(obs_srcs, src_srcs)
#        if np.size(G) != 0:
#            obs_pot += np.dot(G, src_vec)
#            pot = np.dot(G, src_vec)
#    
#    #near field interacitons
#    for src_box_idx in interactions.near_list[obs_box_idx]:
#        src_srcs = my_tree.tree[src_box_idx]
#        src_vec = np.array([src_list[idx].weight for idx in src_srcs])
#        G = interactions.build_G(obs_srcs, src_srcs)
#        if np.size(G) != 0:
#            obs_pot += np.dot(G, src_vec)
#    for i, obs in enumerate(obs_srcs):
#        interactions.potentials[obs] += obs_pot[i]






