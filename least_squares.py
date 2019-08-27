# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 15:03:34 2019

@author: Arthur
"""

import numpy as np

N = 10
ny = 3
nu = 3

#noise = [a1, a2, ... , a500] onde 'a' é aleatório pertencente ao intervalo (-0.2, 0.2)
noise = np.random.uniform(low=-0.2, high=0.2, size=N)

#u = [a1, a2, ... , a500] onde 'a' é aleatório pertencente ao intervalo (-1, 1)
u = np.random.uniform(low=-1, high=1, size=N)

#y = [0, 0, 0]
y = np.empty((0,1))
y = np.append(y, ny*[0])


#geração de saída
for k in range(ny, N):
    y = np.append(y, -0.5*y[k-1] - 0.3*y[k-2] + 0.09*y[k-3] + 8.3*u[k-1] + 1.7*u[k-2] - 5.2*u[k-3])

#aplicação de ruído
noisy_y = y + noise

#geração da matriz Phi - matriz de medidas (observações)
j = nu-1
phi = np.empty((0,6))
for i in range(ny-1, N-1):
    line = np.array([y[i], y[i-1], y[i-2], u[j], u[j-1], u[j-2]])
    j += 1
    phi = np.append(phi, [line], axis = 0)
# construindo a matriz Teta_estimado - matriz com os coeficientes da entrada e da saída
#Phi'
phi_transpose = phi.transpose()
#Phi'*Phi
product_phi_transpose_phi = phi_transpose.dot(phi)
#inv(phi'*phi), pptp(product_phi_transpose_phi)
product_phi_transpose_phi_inv = np.linalg.inv(product_phi_transpose_phi)
#inv(Phi'*Phi)*Phi'
teta_est = product_phi_transpose_phi_inv.dot(phi_transpose)
#inv(Phi'*Phi)*Phi'*y(ny+1:end)
teta_est = teta_est.dot(y[ny:])

############ Ordem 1 #################
N1 = 10
ny1 = 1
nu1 = 1

j = nu1-1
phi1 = np.empty((0, nu1 + ny1))
for i in range(ny1-1, N1-1):
    line = np.array([noisy_y[i], u[j]])
    j += 1
    phi1 = np.append(phi1, [line], axis = 0)

phi1_transpose = phi1.transpose()
#phi1'*phi1
product_phi1_phi1_transpose = phi1_transpose.dot(phi1)
#inv(phi'*phi), ppt1phi_1(product_phi_1t_phi_1)
product_phi1_transpose_phi1_inv = np.linalg.inv(product_phi1_phi1_transpose)
#inv(phi1'*phi1)*phi1'
teta1_est = product_phi1_transpose_phi1_inv.dot(phi1_transpose)
#inv(phi1'*phi1)*phi1'*y(ny+1:end)
teta1_est = teta1_est.dot(noisy_y[ny1:])

y_est1 = phi1.dot(teta1_est)
y_est1 = np.append(noisy_y[0:ny1], y_est1, axis = 0)


erro1 = noisy_y - y_est1
MSE1 = 0
for i in range(len(erro1)):
    MSE1 += erro1[i]**2

############ Ordem 2 #################
N2 = 10
ny2 = 2
nu2 = 2

j = nu2-1
phi2 = np.empty((0, nu2 + ny2))
for i in range(ny2-1, N2-1):
    line = np.array([noisy_y[i], noisy_y[i-1], u[j], u[j - 1]])
    j += 1
    phi2 = np.append(phi2, [line], axis = 0)

phi2_transpose = phi2.transpose()
#phi2'*phi2
product_phi2_phi2_transpose = phi2_transpose.dot(phi2)
#inv(phi2'*phi2), ppt2phi2(product_phi2_transpose_phi2)
product_phi2_transpose_phi2_inv = np.linalg.inv(product_phi2_phi2_transpose)
#inv(phi2'*phi2)*phi2'
teta2_est = product_phi2_transpose_phi2_inv.dot(phi2_transpose)
#inv(phi2'*phi2)*phi2'*y(ny+1:end)
teta2_est = teta2_est.dot(noisy_y[ny2:])

y_est2 = phi2.dot(teta2_est)
y_est2 = np.append(noisy_y[0:ny2], y_est2, axis = 0)

erro2 = noisy_y - y_est2

MSE2 = 0
for i in range(len(erro2)):
    MSE2 += erro2[i]**2


############ Ordem 3 #################
N3 = 10
ny3 = 3
nu3 = 3

j = nu3-1
phi3 = np.empty((0, nu3 + ny3))
for i in range(ny3-1, N3-1):
    line = np.array([noisy_y[i], noisy_y[i-1], noisy_y[i-2], u[j], u[j - 1], u[j - 2]])
    j += 1
    phi3 = np.append(phi3, [line], axis = 0)

phi3_transpose = phi3.transpose()
product_phi3_phi3_transpose = phi3_transpose.dot(phi3)
product_phi3_transpose_phi3_inv = np.linalg.inv(product_phi3_phi3_transpose)
teta3_est = product_phi3_transpose_phi3_inv.dot(phi3_transpose)
teta3_est = teta3_est.dot(noisy_y[ny3:])

y_est3 = phi3.dot(teta3_est)
y_est3 = np.append(noisy_y[0:ny3], y_est3, axis = 0)

erro3 = noisy_y - y_est3

MSE3 = 0
for i in range(len(erro3)):
    MSE3 += erro3[i]**2
############ Ordem 4 #################
N4 = 10
ny4 = 4
nu4 = 4

j = nu4-1
phi4 = np.empty((0, nu4 + ny4))
for i in range(ny4-1, N4-1):
    line = np.array([noisy_y[i], noisy_y[i-1], noisy_y[i-2], noisy_y[i-3], u[j], u[j - 1], u[j - 2], u[j - 3]])
    j += 1
    phi4 = np.append(phi4, [line], axis = 0)

phi4_transpose = phi4.transpose()
product_phi4_phi4_transpose = phi4_transpose.dot(phi4)
product_phi4_transpose_phi4_inv = np.linalg.inv(product_phi4_phi4_transpose)
teta4_est = product_phi4_transpose_phi4_inv.dot(phi4_transpose)
teta4_est = teta4_est.dot(noisy_y[ny4:])

y_est4 = phi4.dot(teta4_est)
y_est4 = np.append(noisy_y[0:ny4], y_est4, axis = 0)

erro4 = noisy_y - y_est4
MSE4 = 0
for i in range(len(erro4)):
    MSE4 += erro4[i]**2

############ Ordem 5 #################
N5 = 10
ny5 = 5
nu5 = 5

j = nu5-1
phi5 = np.empty((0, nu5 + ny5))
for i in range(ny5-1, N5-1):
    line = np.array([noisy_y[i], noisy_y[i-1], noisy_y[i-2], noisy_y[i-3], noisy_y[i-4], u[j], u[j - 1], u[j - 2], u[j - 3], u[j - 4]])
    j += 1
    phi5 = np.append(phi5, [line], axis = 0)

phi5_transpose = phi5.transpose()
product_phi5_phi5_transpose = phi5_transpose.dot(phi5)
product_phi5_transpose_phi5_inv = np.linalg.inv(product_phi5_phi5_transpose)
teta5_est = product_phi5_transpose_phi5_inv.dot(phi5_transpose)
teta5_est = teta5_est.dot(noisy_y[ny5:])

y_est5 = phi5.dot(teta5_est)
y_est5 = np.append(noisy_y[0:ny5], y_est5, axis = 0)

erro5 = noisy_y - y_est5

MSE5 = 0
for i in range(len(erro5)):
    MSE5 += erro5[i]**2









