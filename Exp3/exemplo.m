clc; clear all; close all;

N = 500;
ny = 3;
nu = 3;

%ruido = [a1, a2, ... , a500], onde 'a' é aleatório pertencente ao intervalo
%(-0.2, 0.2)
ruido = (-1 + (1+1).*rand(N,1))*0.2;

%u = [a1, a2, ... , a500], onde 'a' é aleatório pertencente ao intervalo
%(-1, 1)
u = (-1 + (1+1).*rand(N,1));

%y = [0,0,0]
y(1:ny) = 0;

for k = ny+1:N
    y(k) = -0.5*y(k-1) - 0.3*y(k-2) + 0.09*y(k-3) + 8.3*u(k-1) + 1.7*u(k-2) - 5.2*u(k-3);
end

y = y';

y_ruidoso_1 = y + ruido;

P = [y(ny:N-1) y(ny-1:N-2) y(ny-2:N-3) u(nu:N-1) u(nu-1:N-2) u(nu-2:N-3)];

T = inv(P'*P)*P'*y(ny+1:end);

%ordem 1

P1 = [y_ruidoso_1(1:N-1) u(1:N-1)];

T1 = inv(P1'*P1)*P1'*y_ruidoso_1(2:end);

y_est1 = P1 * T1;

y_est1 = [y_ruidoso_1(1:1); y_est1];

erro1 = y_ruidoso_1 - y_est1;

MSE1 = sum(erro1.^2);

%ordem 2

P2 = [y_ruidoso_1(2:N-1) y_ruidoso_1(1:N-2) u(2:N-1) u(1:N-2)];

T2 = inv(P2'*P2)*P2'*y_ruidoso_1(3:end);

y_est2 = P2 * T2;

y_est2 = [y_ruidoso_1(1:2); y_est2];

erro2 = y_ruidoso_1 - y_est2;

MSE2 = sum(erro2.^2);

%ordem 3

P3 = [y_ruidoso_1(ny:N-1) y_ruidoso_1(ny-1:N-2) y_ruidoso_1(ny-2:N-3) u(nu:N-1) u(nu-1:N-2) u(nu-2:N-3)];

T3 = inv(P3'*P3)*P3'*y_ruidoso_1(ny+1:end);

y_est3 = P3 * T3;

y_est3 = [y_ruidoso_1(1:ny); y_est3];

erro3 = y_ruidoso_1 - y_est3;

MSE3 = sum(erro3.^2);

%ordem 4

P4 = [y_ruidoso_1(4:N-1) y_ruidoso_1(3:N-2) y_ruidoso_1(2:N-3) y_ruidoso_1(1:N-4) u(4:N-1) u(3:N-2) u(2:N-3) u(1:N-4)];

T4 = inv(P4'*P4)*P4'*y_ruidoso_1(5:end);

y_est4 = P4 * T4;

y_est4 = [y_ruidoso_1(1:4); y_est4];

erro4 = y_ruidoso_1 - y_est4;

MSE4 = sum(erro4.^2);

%ordem 5

P5 = [y_ruidoso_1(5:N-1) y_ruidoso_1(4:N-2) y_ruidoso_1(3:N-3) y_ruidoso_1(2:N-4) y_ruidoso_1(1:N-5) u(4:N-2) u(3:N-3) u(2:N-4) u(1:N-5)];

T5 = inv(P5'*P5)*P5'*y_ruidoso_1(6:end);

y_est5 = P5 * T5;

y_est5 = [y_ruidoso_1(1:5); y_est5];

erro5 = y_ruidoso_1 - y_est5;

MSE5 = sum(erro5.^2);