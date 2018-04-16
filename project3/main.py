import numpy as np
import numpy.linalg as lg
import time
import interaction
import source
import utilities as utils
    
N = 1000

row_tot, col_tot = 10, 10
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
