import numpy as np
import matplotlib.pyplot as plt
import main

fast_times = []
slow_times = []
errors = []

step, modifier = 1, 1
N = np.logspace(1, 4, 20, dtype='int')
for i,n in enumerate(N):
    print("\nIteration: ", i)
    error, slow, fast = main.run(n, step, modifier)
    errors.append(error)
    slow_times.append(slow)
    fast_times.append(fast)
    
plt.loglog(N, fast_times, N, slow_times)
plt.legend(["Fast", "Slow"])
plt.title("Cost vs. N-particles")
plt.ylabel("Time (seconds)")
plt.xlabel("N-particles")
