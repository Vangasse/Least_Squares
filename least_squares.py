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

#geração da matriz P
j = nu-1
P = np.empty((0,6))
for i in range(ny-1, N-1):
    line = np.array([y[i], y[i-1], y[i-2], u[j], u[j-1], u[j-2]])
    j += 1
    P = np.append(P, [line], axis = 0)

################   construindo T    ######################
#P'
P_transpose = P.transpose()
#P'*P
product_P_transpose_P = P_transpose.dot(P)
#inv(P'*P), pptp(product_P_transpose_P)
pptp_inv = np.linalg.inv(product_P_transpose_P)
#inv(P'*P)*P'
T = pptp_inv.dot(P_transpose)
#inv(P'*P)*P'*y(ny+1:end)
T = T.dot(y[ny:])

############ Ordem 1 #################
N1 = 10
ny1 = 1
nu1 = 1

j = nu1-1
P1 = np.empty((0, nu1 + ny1))
for i in range(ny1-1, N1-1):
    line = np.array([noisy_y[i], u[j]])
    j += 1
    P1 = np.append(P1, [line], axis = 0)

P1t = P1.transpose()
#P1'*P1
prod_P1_P1t = P1t.dot(P1)
#inv(P'*P), ppt1p1(product_P1t_P1)
ppt1p1_inv = np.linalg.inv(prod_P1_P1t)
#inv(P1'*P1)*P1'
T1 = ppt1p1_inv.dot(P1t)
#inv(P1'*P1)*P1'*y(ny+1:end)
T1 = T1.dot(noisy_y[ny1:])

y_est1 = P1.dot(T1)
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
P2 = np.empty((0, nu2 + ny2))
for i in range(ny2-1, N2-1):
    line = np.array([noisy_y[i], noisy_y[i-1], u[j], u[j - 1]])
    j += 1
    P2 = np.append(P2, [line], axis = 0)

P2t = P2.transpose()
#P2'*P2
prod_P2_P2t = P2t.dot(P2)

#inv(P'*P), ppt2p2(product_P2t_P2)
ppt2p2_inv = np.linalg.inv(prod_P2_P2t)

#inv(P2'*P2)*P2'
T2 = ppt2p2_inv.dot(P2t)

#inv(P1'*P1)*P1'*y(ny+1:end)
T2 = T2.dot(noisy_y[ny2:])

y_est2 = P2.dot(T2)
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
P3 = np.empty((0, nu3 + ny3))
for i in range(ny3-1, N3-1):
    line = np.array([noisy_y[i], noisy_y[i-1], noisy_y[i-2], u[j], u[j - 1], u[j - 2]])
    j += 1
    P3 = np.append(P3, [line], axis = 0)

P3t = P3.transpose()
prod_P3_P3t = P3t.dot(P3)
ppt3p3_inv = np.linalg.inv(prod_P3_P3t)
T3 = ppt3p3_inv.dot(P3t)
T3 = T3.dot(noisy_y[ny3:])

y_est3 = P3.dot(T3)
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
P4= np.empty((0, nu4 + ny4))
for i in range(ny4-1, N4-1):
    line = np.array([noisy_y[i], noisy_y[i-1], noisy_y[i-2], noisy_y[i-3], u[j], u[j - 1], u[j - 2], u[j - 3]])
    j += 1
    P4 = np.append(P4, [line], axis = 0)

P4t = P4.transpose()
prod_P4_P4t = P4t.dot(P4)
ppt4p4_inv = np.linalg.inv(prod_P4_P4t)
T4 = ppt4p4_inv.dot(P4t)
T4 = T4.dot(noisy_y[ny4:])

y_est4 = P4.dot(T4)
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
P5 = np.empty((0, nu5 + ny5))
for i in range(ny5-1, N5-1):
    line = np.array([noisy_y[i], noisy_y[i-1], noisy_y[i-2], noisy_y[i-3], noisy_y[i-4], u[j], u[j - 1], u[j - 2], u[j - 3], u[j - 4]])
    j += 1
    P5 = np.append(P5, [line], axis = 0)

P5t = P5.transpose()
prod_P5_P5t = P5t.dot(P5)
ppt5p5_inv = np.linalg.inv(prod_P5_P5t)
T5 = ppt5p5_inv.dot(P5t)
T5 = T5.dot(noisy_y[ny5:])

y_est5 = P5.dot(T5)
y_est5 = np.append(noisy_y[0:ny5], y_est5, axis = 0)

erro5 = noisy_y - y_est5

MSE5 = 0
for i in range(len(erro5)):
    MSE5 += erro5[i]**2









