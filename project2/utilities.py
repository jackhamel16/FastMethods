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