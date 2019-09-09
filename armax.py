# -*- coding: utf-8 -*-
"""
ARMAX algorithm

1- Start defining order(n);
2- Program will read from file "dados_1.csv";
3- Number of Samples is set by data;
4- Function arx is called at bottom;
5- DEBUG is set True for testing only.
"""
###################### Order and Samples ######################
n=3
DEBUG = False

import numpy as np
import pandas as pd

def genRegMatrix(u, y, e, n, N):

    #geração da matriz phi
    phi = np.empty((N-n,0))
    #Output columns:
    for j in range(n):
        y_columm = np.array([])
        for k in range(n-(j+1),N-(j+1)):
            y_columm = np.append(y_columm, y[k])
        y_columm = y_columm.reshape(len(y_columm), 1)
        phi = np.hstack((phi, y_columm))
    #Input columns:
    for j in range(n):
        u_columm = np.array([])
        for k in range(n-(j+1),N-(j+1)):
            u_columm = np.append(u_columm, u[k])
        u_columm = u_columm.reshape(len(u_columm), 1)
        phi = np.hstack((phi, u_columm))
    #Error columns:
    for j in range(n):
        e_columm = np.array([])
        for k in range(n-(j+1),N-(j+1)):
            e_columm = np.append(e_columm, e[k])
        e_columm = e_columm.reshape(len(e_columm), 1)

        phi = np.hstack((phi, e_columm))

    return phi

def armax(u, y, n, N):
    e = np.random.uniform(low=-0.05, high= 0.05, size=(N,1))
    for it in range(100):
        MSE = 0
        #Phi modeling
        phi = genRegMatrix(u, y, e, n, N)
        
        ###################### Minimum Squares Procedure ######################
        phi_t = phi.transpose()
        phi_pseudo_inverse = (np.linalg.inv(phi_t.dot(phi))).dot(phi_t)
        theta = phi_pseudo_inverse.dot(y[n:])

        ###################### Model Evaluation ######################
        y_est = phi.dot(theta)
        y_est = np.append(y[0:n], y_est, axis=0)

        erro = y - y_est
        e = erro

        for i in range(len(erro)):
            MSE += erro[i] ** 2

        return theta

###################### File Reading ######################
data = pd.read_csv('dados_1.csv')
data_matrix = data.to_numpy()
if not DEBUG:
    y = data_matrix[:,0]
u = data_matrix[:,1]

###################### Known Model Testing ######################
if DEBUG:
    y = np.empty((0,1))
    y = np.append(y, n*[0])
    for k in range(n, len(u)):
        y = np.append(y, -0.5*y[k-1] - 0.3*y[k-2] + 0.09*y[k-3] + 8.3*u[k-1] + 1.7*u[k-2] - 5.2*u[k-3])

###################### Resulting Array of Parameters ######################
theta = armax(u,y,n, len(y))
print(theta)
    
