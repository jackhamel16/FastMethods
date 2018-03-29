import numpy as np
import numpy.linalg as lg
import tree
import interaction
import utilities as utils


leaf_start = 2**(2*(level_cnt-1))
leaf_end = 2*leaf_start
parent_start = leaf_start / 4
parent_end = leaf_end / 4

for obs_idx in range(leaf_start, leaf_end):
    count = 0
    if len(my_tree.tree[obs_idx]) == 0:
        continue
    for src_idx in interactions.list[obs_idx]:
        if len(my_tree.tree[src_idx]) == 0:
            continue
        count += 1
        G = interactions.build_G(my_tree.tree[obs_idx], my_tree.tree[src_idx])
        U,V = utils.uv_decompose(G,1)
        interactions.uv_list[obs_idx].append((U,V))
    print('obs: ', obs_idx, '| interaction count: ', count)
        
test2 = my_tree.tree
test = interactions.uv_list
#G = interactions.build_G(my_tree.tree[64], my_tree.tree[68])
#
#G1, G2 = G[0:1,:], G[1:,:]
#
#U1,V1 = utils.uv_decompose(G1,1)
#U2,V2 = utils.uv_decompose(G2,1)
#
#U12,V12 = utils.merge(U1,V1,U2,V2,1)
#
#x = np.array([4,7,4,0])
#print(np.dot(G,x))
#
#Gtest = np.dot(U12,V12)
#print(np.dot(Gtest,x))