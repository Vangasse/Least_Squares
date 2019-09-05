%% 
clear all; close all; clc;

%% ARX

u = rand(100,1); %Dados de entrada do sistema
y = zeros(100,1);
order = 3;
N = length(y); %Quantidade de observa��es

for i = order+1:N
    y(i) = -0.5*y(i-1) - 0.3*y(i-2) + 0.09*y(i-3) + 8.3*u(i-1) + 1.7*u(i-2) - 5.2*u(i-3);
end

lambda = 0.97;

phi = montaMatReg(order, y, u, true, 0); % Matriz de regressores

P = eye(order*2)*10000;
theta = rand(order*2,1);
thetas = theta;

for k = 1:N-order
    K = P*phi(k,:)'/(phi(k,:)*P*phi(k,:)'+lambda);
    theta = theta + K*(y(k+order) - phi(k,:)*theta);
    P = (1/lambda) * (P - (P*phi(k,:)'*phi(k,:)*P)/(phi(k,:)*P*phi(k,:)'+lambda));
    thetas = [thetas theta];
    
end