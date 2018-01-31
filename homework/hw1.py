import time
import numpy as np
import matplotlib.pyplot as plt

j0 = np.complex(0,1)

def create_arrays(N):
    n = np.linspace(0, 2*np.pi-2*np.pi/N, N)
    r = np.exp(j0 * n)
    a = np.ones(N) / np.abs([r[i] for i in range(N)] - r[0])
    a[0] = 1
    
    A_mat = np.array([np.zeros(N) for i in range(N)])       
    for i in range(N):
        A_mat[:,i] = a
        a = np.roll(a,1)
        
    return(A_mat, a, np.random.rand(N)) 

def fft_solve(a, x):
    aft, xft = np.fft.fft(a), np.fft.fft(x)
    return(np.fft.ifft(aft*xft))
    
def test(T):
    fft_times, mult_times = np.zeros(T), np.zeros(T)
    error = 0
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

        error += np.linalg.norm(fft_b - mult_b)/np.linalg.norm(mult_b)
        
    error = error / T
    
    return(N_list, np.array([fft_times,mult_times]), error, fft_b, mult_b)
    

iters = 10
T = 140
times = np.array([np.zeros(T), np.zeros(T)])
error = 0
for i in range(iters):
    N, time_out, error_out, fft_b, mult_b = test(T)
    times += time_out
    error += error_out

error, times = error / iters, times / iters

plt.loglog(N, times[0], label="FFT Solve")
plt.loglog(N, times[1], label="MatVec Product")
plt.loglog(N, N*np.log10(N) * 1e-7, label="Normalized Nlog(N)")
plt.legend(loc="upper left")
plt.title("Matrix Vector Product: FFT Solve vs. Multiplication")
plt.ylabel("log(t)")
plt.xlabel("N")
