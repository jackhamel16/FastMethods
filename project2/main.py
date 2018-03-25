import numpy as np
import source
import tree

level_cnt = 4 # Count levels starting from root = 0 
grid_dim = 2**(level_cnt-1) # Should remain power of two for easy life
grid_step = 1
N = 5

src_list = []
#src_list = [source.source(4.5,1.5,1), source.source(1.2,2.3,1), \
#            source.source(7,7.1,1)]
for i in range(N):
    src_list.append(source.source(grid_dim * np.random.random(),grid_dim * \
                              np.random.random(), np.random.random()))
    # Map src to nearest lower left grid pnt
    src_list[i].grid = (int(np.floor(src_list[i].x/grid_step)), \
                        int(np.floor(src_list[i].y/grid_step)))
    
linear_tree = tree.tree(src_list, level_cnt)
linear_tree = linear_tree.build()

interactions = tree.interaction(level_cnt)
interactions.fill_child(2,1,level_cnt-2)
interactions.fill_list()












