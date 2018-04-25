import numpy as np

def idx2coord(i, c):
    row = np.floor(i/c)
    col = i%c
    return(int(row), int(col))

def coord2idx(r, c, ctot):
    return(int(c + r * ctot))

def cart2cyl(x, y):
    if (x == 0):
        if (y == 0):
            theta = 0
        else:
            theta = np.pi * y / (2 * np.abs(y))
    if (x > 0):
        theta = np.arctan(y / x)
    if (x < 0) and (y >= 0):
        theta = np.arctan(y / x) + np.pi
    if (x < 0) and (y <= 0):
        theta = np.arctan(y / x) - np.pi
    rho = np.sqrt(x**2 + y**2)
    return(rho, theta)