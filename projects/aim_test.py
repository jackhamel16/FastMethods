import matplotlib.pyplot as plt
import numpy as np

import proj1 as aim
#
#N =  [i*50 for i in range(5,60)]
#step = 0.05
#avg_error = 0
#fast_times = []
#slow_times = []
#for n in N:
#    D = (np.ceil(np.sqrt(n)/2))
#    time_fast, time_slow, error = aim.run(n, D, D, step)
#    avg_error += error/len(N)
#    fast_times.append(time_fast)
#    slow_times.append(time_slow)
#
#print("Average Error: ", avg_error)
#plt.close()
#plt.semilogy(N, fast_times, N, slow_times)
#plt.legend(("Fast", "Slow"))
#plt.title("AIM vs. Direct")
#plt.xlabel("N Particles")
#plt.ylabel("Time (seconds)")
#
#N = 100
#steps = np.arange(0.005,0.45,0.005)
#errors= []
#for step in steps:
#    D = 4
#    time_fast, time_slow, error = aim.run(n, D, D, step)
#    errors.append(error)
#
#print("Average Error: ", avg_error)
#plt.figure()
#plt.plot(steps, errors)
#plt.title("Error vs. Step Size")
#plt.xlabel("Step Size")
#plt.ylabel("Error")

N = 100
sizes = np.arange(1,15.25,0.25)
step = 0.05
errors= []
for size in sizes:
    time_fast, time_slow, error = aim.run(N, size, size, step)
    errors.append(error)

print("Average Error: ", avg_error)
plt.figure()
plt.plot(sizes, errors)
plt.title("Error vs. Source Density")
plt.xlabel("Size of Grid Dimensions")
plt.ylabel("Error")

#time_file = open("timings.dat", "w")
#print("{0:^15s} {1:^15s} {2:^15s}".format("N", "Fast", "Slow"), file=time_file)
#for i in range(len(N)):
#    print("{0:^15d} {1:4.11f} {2:4.11f}"\
#          .format(N[i], fast_times[i], slow_times[i]), file = time_file)
#time_file.close()