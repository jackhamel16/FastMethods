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