import numpy as np
import matplotlib.pyplot as plt
import main

#size = 30
#N = np.logspace(1, 4.3, size)
#lvl_cnt = 4
#grid_step = 1
#eps = 1e-1
#fast, slow, error = main.run(lvl_cnt, grid_step, 10, eps)
#fast_vec = np.zeros(size)
#slow_vec = np.zeros(size)
#error_vec = np.zeros(size)
#for i,n in enumerate(N):
#    print('\nN = ', int(np.ceil(n)))
#    fast, slow, error = main.run(lvl_cnt, grid_step, int(np.ceil(n)), eps)
#    fast_vec[i] = fast
#    slow_vec[i] = slow
#    error_vec[i] = error
#    
#plt.loglog(N, fast_vec, N, slow_vec)
#plt.legend(['Fast Time', 'Slow Time'])
#plt.xlabel('N-particles')
#plt.ylabel('Time (sec)')
#plt.title('Cost Comparison of Using a Rank Deficient Algorithm')

size = 30
eps = np.logspace(-10, 1, size)
lvl_cnt = 4
grid_step = 1
N = 1000
fast_vec = np.zeros(size)
slow_vec = np.zeros(size)
error_vec = np.zeros(size)
for i,e in enumerate(eps):
    print('\neps = ', e)
    fast, slow, error = main.run(lvl_cnt, grid_step, N, e)
    fast_vec[i] = fast
    slow_vec[i] = slow
    error_vec[i] = error
    
#plt.loglog(eps, fast_vec)
#plt.legend(['Fast Time'])
#plt.xlabel('Epsilon')
#plt.ylabel('Time (sec)')
#plt.title('Cost Effect of Increasing Approximation Error')
    
plt.loglog(eps, np.abs(error_vec))
plt.xlabel('Epsilon')
plt.ylabel('Error')
plt.title('Error Effect of Increasing Approximation Tolerance')