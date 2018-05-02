import matplotlib.pyplot as plt
import main

fast_times = []
slow_times = []
errors = []

step, modifier = 1, 1
N = np.logspace(1, 4, 20, dtype='int')
for n in N:
    error, slow, fast = main.run(n, step, modifier)
    errors.append(error)
    slow_times.append(slow)
    fast_times.append(fast)
    
plt.semilogx(N, fast, N, slow)
plt.legend("Fast", "Slow")

