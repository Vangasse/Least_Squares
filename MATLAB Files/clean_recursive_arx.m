%% 
clear all; close all; clc;

%% ARX


%%
data = load('dados_1.txt');
u = data(:,2); %Dados de entrada do sistema
y = data(:,1);
order = 3;
N = length(y); %Quantidade de observa��es

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