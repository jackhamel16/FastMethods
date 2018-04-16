import numpy as np

def idx2coord(i, c):
    row = np.floor(i/c)
    col = i%c
    return(int(row), int(col))

def coord2idx(r, c, ctot):
    return(int(c + r * ctot))