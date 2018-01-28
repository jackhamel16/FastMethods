import numpy as np

N = 10
n = np.linspace(0, 2*np.pi-2*np.pi/N, N)
x,y = np.array(cos(n)), np.array(sin(n))

A = np.array([zeros(N) for i in range(N)])
for i in range(N):
    for j in range(N):
        if (i==j):
            A[i,j] = 1
            continue
        x_dist, y_dist = abs(x[i]-x[j]), abs(y[i]-y[j])
        A[i,j] = 1 / np.sqrt(x_dist**2+y_dist**2)
        
#def test1(A,x,N):
    



x = np.random.rand(N)
a = np.array(A[:,0])
aft, xft = np.fft.fft(a), np.fft.fft(x)
b = fft.ifft(np.inner(np.diag(aft),xft))


#Aft, xft = np.fft.fft2(A), np.fft.fft(x)
#b = np.fft.ifft2(np.inner(np.diag(xft),Aft))

b2 = np.inner(A,x)
