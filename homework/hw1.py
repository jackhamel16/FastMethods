import numpy as np

N = 1000
n = np.linspace(0, 2*np.pi-2*np.pi/N, N)
x,y = np.array(np.cos(n)), np.array(np.sin(n))

A = np.matrix([np.zeros(N) for i in range(N)])
for i in range(N):
    for j in range(N):
        x_dist, y_dist = abs(x[i]-x[j]), abs(y[i]-y[j])
        if (i==j):
            A[i,j] = 1
            continue
        A[i,j] = 1 / np.sqrt(x_dist**2+y_dist**2)
        

x = np.ones(N)
xft = np.fft.fft(x)
Aft = np.fft.fft2(A)
b = np.fft.ifft2(np.inner(diag(xft),Aft))

b2 = np.inner(A,x)
