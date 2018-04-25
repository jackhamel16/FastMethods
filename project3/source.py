import numpy as np

class source:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.rho = "not a num" #fill with c2m rho
        self.theta = "not a num" # fill with c2m theta
        self.weight = weight
        self.grid = np.array([0,0]) # fill with grid point it maps to 
        self.idx = "not a num" # fill with idx of grid point
        
    def calculate_theta(self):
        if (self.x == 0):
            return(np.pi * self.y / (2 * np.abs(self.y)))
        if (self.x > 0):
            return(np.arctan(self.y / self.x))
        if (self.x < 0) and (self.y >= 0):
            return(np.arctan(self.y / self.x) + np.pi)
        if (self.x < 0) and (self.y <= 0):
            return(np.arctan(self.y / self.x) - np.pi)
        
#    def find_distance(src1, src2):
#        x = np.sqrt(src2.x - src1.x)
#        y = np.sqrt(src2.y - src1.y)
#        rho = np.sqrt(x**2 + y**2)
#        if (x >= 0):
#            theta = np.arctan(y / x)
#        if (x <= 0) and (y >= 0):
#            theta = np.arctan(y / x) + np.pi
#        if (x <= 0) and (y <= 0):
#            theta = np.arctan(y / x) - np.pi
#        return
