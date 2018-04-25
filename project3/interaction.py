import numpy as np
import utilities as utils
import scipy.special as funcs

class interaction:
    def __init__(self, box_tot, col_tot, row_tot, src_list, box_list):
        self.box_tot = box_tot
        self.row_tot = row_tot
        self.col_tot = col_tot
        self.list = [[] for i in range(box_tot)]
        self.near_list = [[] for i in range(box_tot)]
        self.src_list = src_list
        self.box_list = box_list
        self.potentials = np.zeros(len(self.src_list))
        
    def fill_lists(self):
        for obs_box_idx in range(self.box_tot):
            orow, ocol = utils.idx2coord(obs_box_idx, self.col_tot)
            near_row_range = np.array([-1, 0, 1]) + orow
            near_col_range = np.array([-1, 0, 1]) + ocol
            for src_box_idx in range(self.box_tot):
                srow, scol = utils.idx2coord(src_box_idx, self.col_tot)
                near_flag = 0
                for r in near_row_range:
                    if (r < 0) or (r >= self.row_tot):
                        continue
                    for c in near_col_range:
                        if (c < 0) or (c >= self.col_tot):
                            continue
                        if (scol == c) and (srow == r):
                            near_flag = 1
                            self.near_list[obs_box_idx].append(src_box_idx)
                if (near_flag == 0):
                    self.list[obs_box_idx].append(src_box_idx)                    
                    
    def build_G(self, obs_idxs, src_idxs, k):
        G = np.array([np.zeros(len(src_idxs), dtype='complex') \
                      for i in range(len(obs_idxs))])
        obs_srcs = [self.src_list[idx] for idx in obs_idxs]
        src_srcs = [self.src_list[idx] for idx in src_idxs]
        for i, obs in enumerate(obs_srcs):
            for j, src in enumerate(src_srcs):
                if (obs.x == src.x) and (obs.y == src.y):
                    continue
                G[i, j] = funcs.hankel1(0, k * np.sqrt((obs.x - src.x)**2 + \
                                                        (obs.y - src.y)**2))
        return(G)
    
    def compute_near(self, obs_box_idx, k):
        obs_idxs = self.box_list[obs_box_idx]
        obs_pot = np.zeros(len(obs_idxs), dtype='complex')
        src_box_list = self.near_list[obs_box_idx]
        for src_box_idx in src_box_list:
            src_idxs = self.box_list[src_box_idx]
            src_vec = np.array([self.src_list[idx].weight for idx in src_idxs])
            G = self.build_G(obs_idxs, src_idxs, k)
            if np.size(G) != 0:
                obs_pot += np.dot(G, src_vec)
        return(obs_pot)
    