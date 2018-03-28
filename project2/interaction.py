import numpy as np
import utilities as utils

class interaction:
    def __init__(self, level_cnt, tree):
        self.level_cnt = level_cnt
        self.list = [[] for i in range(2**(level_cnt*2-1))]
        self.near_list = [[] for i in range(2**(level_cnt*2-1))]
        self.my_tree = tree
        self.src_list = tree.src_list
        self.potentials = np.zeros(len(self.src_list))
        
    def fill_child(self, cx, cy, child_lvl):
        child_id = utils.pnt2idx(cx, cy , child_lvl)
        parent_lvl = child_lvl - 1
        parent_id = child_id[0:-2]
        p_dim = 2**parent_lvl
        px, py = utils.idx2pnt(parent_id)
        x_range, y_range = px + np.array([-1,0,1]), py + np.array([-1,0,1])
        for x in x_range:
            if x >= 0 and x <= p_dim-1:
                for y in y_range:
                    if y >= 0 and y <= p_dim-1 and (px != x or py != y):
                        pn_id = utils.pnt2idx(x, y, parent_lvl)
                        cf_id_start = int(pn_id + '00', 2)
                        for cf_id in range(cf_id_start, cf_id_start+4):
                            cf_x, cf_y = utils.idx2pnt('0' + \
                                                 format(cf_id, '0'+\
                                                        str(child_lvl)+'b'))
                            if (abs(cx-cf_x) > 1) or (abs(cy-cf_y) > 1):
                                self.list[int(child_id,2)].append(cf_id)
                            if (abs(cx-cf_x) < 2) and (abs(cy-cf_y) < 2) and \
                                             child_lvl == (self.level_cnt - 1):
                                self.near_list[int(child_id,2)].append(cf_id)
        if child_lvl == (self.level_cnt - 1):
            cn_id_start = int(parent_id + '00', 2)
            for cn_id in range(cn_id_start, cn_id_start+4):
                self.near_list[int(child_id,2)].append(cn_id)
    
    def fill_list(self):
        for child_lvl in range(self.level_cnt-1, 1, -1):
            child_dim = 2**child_lvl
            for cx in range(child_dim):
                for cy in range(child_dim):
                    self.fill_child(cx, cy, child_lvl)
                    
    def build_G(self, obs_srcs, src_srcs):
        G = np.array([np.zeros(len(src_srcs)) for i in range(len(obs_srcs))])
        for i, obs in enumerate(obs_srcs):
            for j, src in enumerate(src_srcs):
                if (self.src_list[obs].x == self.src_list[src].x) and \
                   (self.src_list[obs].y == self.src_list[src].y):
                    continue
                G[i, j] = 1 / np.sqrt((self.src_list[obs].x - \
                          self.src_list[src].x)**2 + (self.src_list[obs].y - \
                          self.src_list[src].y)**2)
        return(G)
    
    def compute_box_pot_slow(self, obs_box_idx, near=0):
        obs_srcs = self.my_tree.tree[obs_box_idx]
        obs_pot = np.zeros(len(obs_srcs))
        if near == 0:
            box_list = self.list[obs_box_idx]
        else:
            box_list = self.near_list[obs_box_idx]
        for src_box_idx in box_list:
            src_srcs = self.my_tree.tree[src_box_idx]
            src_vec = np.array([self.src_list[idx].weight for idx in src_srcs])
            G = self.build_G(obs_srcs, src_srcs)
            if np.size(G) != 0:
                obs_pot += np.dot(G, src_vec)
        return(obs_pot)
    