import timeit
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
    return(np.fft.ifft(np.inner(np.diag(aft),xft)))

def test(T):
    fft_times, mult_times = np.zeros(T), np.zeros(T)

    for i in range(1,T):
        N = 100*i
        A, a, x = create_arrays(N)
                
        fft_start = timeit.timeit()
        fft_b = fft_solve(a, x)
        fft_end = timeit.timeit()
        fft_times[i] = fft_end - fft_start

        mult_start = timeit.timeit()
        mult_b = np.inner(A, x)
        mult_end = timeit.timeit()
        mult_times[i] = mult_end - mult_start
    return(N,tuple([fft_times,mult_times]))
    

N, times = test(40)
       
#Aft, xft = np.fft.fft2(A), np.fft.fft(x)
#b = np.fft.ifft2(np.inner(np.diag(xft),Aft))
N = 10
A, a, x = create_arrays(N)

b = fft_solve(a, x)
b2 = np.inner(A,x)
