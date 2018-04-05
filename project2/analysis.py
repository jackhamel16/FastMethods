import numpy as np
import matplotlib.pyplot as plt
import main

size = 20
N = np.logspace(1, 3, size)
lvl_cnt = 4
grid_step = 1
eps = 1e-4
fast, slow, error = main.run(lvl_cnt, grid_step, 10, eps)
fast_vec = np.zeros(size)
slow_vec = np.zeros(size)
error_vec = np.zeros(size)
for i,n in enumerate(N):
    print('\nN = ', int(np.ceil(n)))
    fast, slow, error = main.run(lvl_cnt, grid_step, int(np.ceil(n)), eps)
    fast_vec[i] = fast
    slow_vec[i] = slow
    error_vec[i] = error
    
plt.loglog(N, fast_vec, N, slow_vec)
    