import utilities as utils

class tree:
    def __init__(self, src_list, level_count):
        self.src_list = src_list
        self.levels = level_count
        self.tree = [[] for i in range(2**(self.levels*2-1))]
    
    def build(self):
        # Returns tree as list of lists containing srcs in box 
        for i,src in enumerate(self.src_list):
            src.idx = utils.pnt2idx(src.grid[0],src.grid[1],self.levels-1)
            for level in range(2,self.levels+1):
                self.tree[int(src.idx[0:2*level],2)].append(i)
                
    def get_leaves(self, idx, lvl):
        if type(idx) == int:
            static_bits = utils.binary(idx, lvl)
        else:
            static_bits = idx
        leaves = []
        level_diff = (self.levels)-lvl-1
        for i in range(2**(2*level_diff)):
            dynamic_bits = format(i, '0'+str(2*level_diff)+'b')
            leaves.append(int(static_bits+dynamic_bits, 2))
        return(leaves)
    
    def get_children(self, idx, lvl):
        if type(idx) == int:
            static_bits = utils.binary(idx, lvl)
        else:
            static_bits = idx
        children = []
        for i in range(4):
            dynamic_bits = format(i, '02b')
            children.append(int(static_bits+dynamic_bits, 2))
        return(children)
    