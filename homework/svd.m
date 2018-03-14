m = 100;
A = ones(m,m);

box1_x = rand(1,m);
box1_y = rand(1,m);
box2_x = box1_x + 100;
box2_y = box1_y;

for i=1:m
    for j=1:m
        A(i,j) = 1/sqrt((box1_x(i)-box2_x(j))^2 + (box1_y(i)-box2_y(j))^2);
    end
end

x = rand(m,1);
[U,S,V] = svd(A);
b = A * x;

Vt = V';

N = 100;
A_error_vec = zeros(1,N);
b_error_vec = zeros(1,N);
for n=1:N
    U1 = U(:,1:n);
    S1 = S(1:n,1:n);
    V1t = Vt(1:n,:);

    A_app = U1 * S1 * V1t;
    b_app = A_app * x;
    
    b_error_vec(n) = norm(b_app-b)/norm(b);
    A_error_vec(n) = norm(A_app-A)/norm(A);
end
close all
semilogy([1:N], b_error_vec)

