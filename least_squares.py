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

'''
print(P)
print(T)
print(y)









