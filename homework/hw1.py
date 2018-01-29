import time
import numpy as np
import matplotlib.pyplot as plt

def create_arrays(N):
    n = np.linspace(0, 2*np.pi-2*np.pi/N, N)
    x,y = np.array(np.cos(n)), np.array(np.sin(n))
    
    A_mat = np.array([np.zeros(N) for i in range(N)])
    a = np.ones(N)
    for i in range(1,N):
        x_dist, y_dist = abs(x[i]-x[0]), abs(y[i]-y[0])
        a[i] = 1 / np.sqrt(x_dist**2+y_dist**2)
    
    x_vec = np.random.rand(N)        
    A_mat = np.array([a for i in range(N)])
    return(A_mat,a,x_vec) 

def fft_solve(a, x):
    aft, xft = np.fft.fft(a), np.fft.fft(x)
    return(np.fft.ifft(aft*xft))
    
def test(T):
    fft_times, mult_times = np.zeros(T), np.zeros(T)
    N_list = []

    for i in range(1,T+1):
        N = 20*i
        N_list.append(N)
        A, a, x = create_arrays(N)
                
        fft_start = time.time()
        fft_b = fft_solve(a, x)
        fft_end = time.time()
        fft_times[i-1] = fft_end - fft_start

        mult_start = time.time()
        mult_b = np.inner(A, x)
        mult_end = time.time()
        mult_times[i-1] = mult_end - mult_start
    return(N_list,np.array([fft_times,mult_times]))
    
def fft_solve2(a, x):
    aft, xft = np.fft.fft(a), np.fft.fft(x)
    return(np.fft.ifft(aft*xft))
    
iters = 10
T = 120
times = np.array([np.zeros(T), np.zeros(T)])
for i in range(iters):
    N, time_out = test(T)
    times += time_out
    
times = times / iters

plt.loglog(N, times[0], label="FFT Solve")
plt.loglog(N, times[1], label="MatVec Product")
plt.legend(loc="upper left")
plt.title("Matrix Vector Product: FFT Solve vs. Multiplication")
plt.ylabel("log(t)")
plt.xlabel("log(N)")
