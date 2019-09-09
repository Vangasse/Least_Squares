# -*- coding: utf-8 -*-
"""
Recursive ARX algorithm
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
import time

def genRegMatrix(u, y, n, N):
    N = len(y)
    ###################### Phi Construction ######################
    # Inicialização da matriz phi
    phi = np.empty((N-n,0))
    """
    Neste algoritmo, as colunas da matriz phi são geradas incrementalmente
    de acordo com seus limites e adicionadas à mesma através da função
    hstack(numpy).
    """
    #Output columns:
    for j in range(n):
        y_column = np.array([])
        for k in range(n - (j + 1), N - (j + 1)):
            y_column= np.append(y_column, y[k])
        y_column = y_column.reshape(len(y_column), 1)   # Transforma o array linha em coluna
        phi = np.hstack((phi, y_column))                # Adiciona o array coluna na matriz phi
    
    #Input columns:
    for j in range(n):
        u_column = np.array([])
        for k in range(n - (j + 1), N - (j + 1)):
            u_column = np.append(u_column, u[k])
        u_column = u_column.reshape(len(u_column), 1)   # Transforma o array linha em coluna
        phi = np.hstack((phi, u_column))                # Adiciona o array coluna na matriz phi
    
    return phi
    


def recursive_arx(u, y, n, l, N):
    ###################### Setting Parameters ######################
    #Regressor Matrix
    phi = genRegMatrix(u, y, n, N)
    #Covariance Matrix
    P = np.identity(n*2)*10000
    #System Parameters
    theta_est = np.random.uniform(low=-1, high=1, size=(n*2, 1))
    ###################### Recursive Procedure ######################
    """N-n"""
    for i in range(N-n):
        row_phi = phi[i,:].reshape(1, 2*n)
        row_phi_transpose = row_phi.transpose()

        den_K = row_phi.dot(P).dot(row_phi_transpose) + l
        num_K = P.dot(row_phi_transpose)
        K = num_K / den_K

        theta_est = theta_est + K*(y[i+n] - row_phi.dot(theta_est))
        den_P = ((row_phi.dot(P)).dot(row_phi_transpose)) + l
        num_P = ((P.dot(row_phi_transpose)).dot(row_phi)).dot(P)
        P = (1/l)*(P - (num_P/den_P))
    
    return theta_est


###################### File Reading ######################
data = pd.read_csv('dados_1.csv')
data_matrix = data.to_numpy()
if not DEBUG:
    y = data_matrix[:,0]
u = data_matrix[:,1]

###################### Known Model Testing ######################
if DEBUG:
    u = np.random.uniform(size = 10000)
    y = np.empty((0,1))
    y = np.append(y, n*[0])
    for k in range(n, len(u)):
        y = np.append(y, -0.5*y[k-1] - 0.3*y[k-2] + 0.09*y[k-3] + 8.3*u[k-1] + 1.7*u[k-2] - 5.2*u[k-3])

###################### Resulting Array of Parameters ######################
theta = recursive_arx(u,y,n, 0.97, len(y))#lambda = 0.97
print(theta)