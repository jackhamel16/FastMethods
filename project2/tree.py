import numpy as np
import source

class tree:
    def __init__(self, src_list, level_count):
        self.src_list = src_list
        self.levels = level_count
        self.tree = [[] for i in range(2**(self.levels*2-1))]
    
    
    def build(self):
        # Returns tree as list of lists containing srcs in box 
        for i,src in enumerate(self.src_list):
            src.idx = pnt2idx(src.grid[0], src.grid[1], self.levels-1)
            for level in range(2,self.levels+1):
                self.tree[int(src.idx[0:2*level],2)].append(i)
                
class interaction:
    def __init__(self, level_cnt):
        self.level_cnt = level_cnt
        self.list = [[] for i in range(2**(level_cnt*2-1))]
        
    def fill_child(self, cx, cy, child_lvl):
        child_id = pnt2idx(cx, cy ,child_lvl)
        parent_lvl = child_lvl - 1
        parent_id = child_id[0:-2]
        p_dim = 2**parent_lvl
        px, py = idx2pnt(parent_id)
        x_range, y_range = px + np.array([-1,0,1]), py + np.array([-1,0,1])
        for x in x_range:
            if x >= 0 and x <= p_dim-1:
                for y in y_range:
                    if y >= 0 and y <= p_dim-1 and (px != x or py != y):
                        pn_id = pnt2idx(x, y, parent_lvl)
                        cn_id_start = int(pn_id + '00', 2)
                        for cn_id in range(cn_id_start, cn_id_start+4):
                            cn_x, cn_y = idx2pnt('0' + \
                                                 format(cn_id, '0'+\
                                                        str(child_lvl)+'b'))
                            if (abs(cx-cn_x) > 1) or (abs(cy-cn_y) > 1):
                                self.list[int(child_id,2)].append(cn_id)
    
    def fill_list(self):
        for child_lvl in range(self.level_cnt-1, 1, -1):
            child_dim = 2**child_lvl
            for cx in range(child_dim):
                for cy in range(child_dim):
                    self.fill_child(cx, cy, child_lvl)

    
def pnt2idx(x, y, level):
    # Returns index, as a string in binary, of a point using Morton Coding
    # level should be the current level in the tree
    bx = format(x, '0'+str(level)+'b')
    by = format(y, '0'+str(level)+'b')

    if level != 0:
        return('01'+''.join([by[i:i+1]+bx[i:i+1] for i in range(len(bx))]))
    else:
        return('01')
    
def idx2pnt(idx):
    bx, by = '', ''
    for i, bit in enumerate(idx[2:]):
        if i % 2 == 0:
            by += bit
        else:
            bx += bit
    return(int(bx,2), int(by,2))










