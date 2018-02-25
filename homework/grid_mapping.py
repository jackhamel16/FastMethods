import numpy as np
import scipy.linalg as lg
import matplotlib.pyplot as plt
import random

class source:
    def __init__(self, x, y, weight):
        self.x = x
        self.y = y
        self.weight = weight
        self.grid = (0,0)
        
# set up grid
Dx, Dy = 5,5
x_boxes, y_boxes = 5,5
step = Dx/x_boxes
grid = np.array([np.zeros(y_boxes+1) for i in range(x_boxes+1)])
Mrx, Mry = x_boxes, y_boxes
Msp = 1
tau = 4

N = 10
src_list = []
for i in range(N):
    src_list.append(source(Dx * np.random.random(), Dy * np.random.random(),\
                    np.random.random()))

# Map to nearest bottom left corner
for src in src_list:
    src.grid = (int(np.floor(src.x/step)), int(np.floor(src.y/step)))
    E1 = np.exp(-((src.x-src.grid[0])**2+\
                  (src.y-src.grid[1])**2)/(4*tau))
    E2x = np.exp(np.pi*(src.x-src.grid[0])/(Mrx*tau))
    E2y = np.exp(np.pi*(src.y-src.grid[1])/(Mry*tau))
    V0 = src.weight * E1
    for l2 in range(-Msp+1,Msp+1):
        Vy = V0 * E2y**l2
        for l1 in range(-Msp+1,Msp+1):
            grid[src.grid[0]+l1, src.grid[1]+l2] += Vy * E2x**l1
            print(Vy * E2x**l1)


test = np.exp(-((1.2-1)**2+\
                  (1.7-1)**2)/(4*tau))