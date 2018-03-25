import numpy as np

class source:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight
        self.grid = np.array([0,0])
        self.idx = 5225