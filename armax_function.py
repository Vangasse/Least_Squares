import numpy as np
def armax_function(Order, u, y):
    n = Order
    N = len(y)
    e = np.random.uniform(low=-0.05, high=0.05, size=(N, 1))
    best_MSE = float('inf')
    for it in range(100):
        MSE = 0
        # geração da matriz phi
        phi = np.empty((N - n, 0))
        # colunas de y:
        for j in range(n):
            y_columm = np.array([])
            for k in range(n - (j + 1), N - (j + 1)):
                y_columm = np.append(y_columm, y[k])
            y_columm = y_columm.reshape(len(y_columm), 1)
            phi = np.hstack((phi, y_columm))
        for j in range(n):
            u_columm = np.array([])
            for k in range(n - (j + 1), N - (j + 1)):
                u_columm = np.append(u_columm, u[k])
            u_columm = u_columm.reshape(len(u_columm), 1)
            phi = np.hstack((phi, u_columm))
        for j in range(n):
            e_columm = np.array([])
            for k in range(n - (j + 1), N - (j + 1)):
                e_columm = np.append(e_columm, e[k])
            e_columm = e_columm.reshape(len(e_columm), 1)

            phi = np.hstack((phi, e_columm))
        # construindo theta
        phi_t = phi.transpose()
        phi_pseudo_inverse = (np.linalg.inv(phi_t.dot(phi))).dot(phi_t)
        theta = phi_pseudo_inverse.dot(y[n:])

        y_est = phi.dot(theta)
        y_est = np.append(y[0:n], y_est, axis=0)

        erro = y - y_est
        e = erro

        for i in range(len(erro)):
            MSE += erro[i] ** 2

    return theta