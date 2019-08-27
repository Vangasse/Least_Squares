import numpy as np
import pandas as pd

# TODO ler o arquivo dados e criar a função arx(data, N)
N=10
n=3

# Essa parte vai ser removida do código final
u = np.random.uniform(low=-1, high=1, size=(N,1))
y = np.empty((0,1))
y = np.append(y, n*[0])
for k in range(n, N):
    y = np.append(y, -0.5*y[k-1] - 0.3*y[k-2] + 0.09*y[k-3] + 8.3*u[k-1] + 1.7*u[k-2] - 5.2*u[k-3])
y = y.reshape(N,1)
# Até aqui

# geração da matriz phi
phi = np.empty((N-n,0))

# colunas de y:
for j in range(n):
    y_columm = np.array([])
    for k in range(n - (j + 1), N - (j + 1)):
        y_columm = np.append(y_columm, y[k])
    y_columm = y_columm.reshape(len(y_columm), 1)   # Transforma o array linha em coluna
    phi = np.hstack((phi, y_columm))                # Adiciona o array coluna na matriz phi

# colunas de u
for j in range(n):
    u_columm = np.array([])
    for k in range(n - (j + 1), N - (j + 1)):
        u_columm = np.append(u_columm, u[k])
    u_columm = u_columm.reshape(len(u_columm), 1)   # Transforma o array linha em coluna
    phi = np.hstack((phi, u_columm))                # Adiciona o array coluna na matriz phi

# Phi_T
phi_transpose = phi.transpose()
# Phi * Phi_T
product_phi_phi_transpose = phi_transpose.dot(phi)
# (Phi * Phi_T)'
product_phi_transpose_phi_inv = np.linalg.inv(product_phi_phi_transpose)
# Theta = (Phi * Phi_T)' * Phi_T
theta_est = product_phi_transpose_phi_inv.dot(phi_transpose)
# Theta = (Phi * Phi_T)' * Phi_T * Y
theta_est = theta_est.dot(y[n:])

y_est = phi.dot(theta_est)
y_est = np.append(y[0:n], y_est, axis=0)

y_mean = np.mean(y_est)
erro = y - y_est

MSE=0
for i in range(len(erro)):
    MSE += erro[i] ** 2
C=0
for k in range(len(y)):
    C += (y[k] - y_mean) ** 2
COEEF = 1 - MSE/C
