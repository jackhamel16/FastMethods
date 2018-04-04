import numpy as np
import main

N = np.logspace(1, 3, 10)
lvl_cnt = 4
grid_step = 1
eps = 1e-1
fast, slow, error = main.run(lvl_cnt, grid_step, 10, eps)
fast_vec = np.zeros(50)
slow_vec = np.zeros(50)
error_vec = np.zeros(50)
for i,n in enumerate(N):
    fast, slow, error = main.run(lvl_cnt, grid_step, int(np.ceil(n)), eps)
    fast_vec[i] = fast
    slow_vec[i] = slow
    error_vec[i] = error
    

    