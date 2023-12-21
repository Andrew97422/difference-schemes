import math
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(projection='3d')

def phi(x, t):
    return np.log(4 + x) * np.cos(5 * t)


def psi(x):
    return 3 - 2 * x


def ua(t):
    return 2 * np.cos(t)


def ub(t):
    return np.sin(t)


A = 0.5
B = 1.5
p = 1.6
T = 1.2

def solve(N, M):
    h = (B - A) / M
    tau = T / N
    q = p * tau / (h ** 2)
    U = []
    
    for i in range(N + 1):
        U.append([])
        for j in range(M + 1):
            U[i].append(0)

    for i in range(M + 1):
        U[0][i] = psi(A + i * h)
    
    alpha = [0] * (M + 1)
    beta = [0] * (M + 1)
    print(len(U), N, M)
    for i in range(N):
        a = [-q] * (M + 1)
        b = [1 + 2 * q] * (M + 1)
        c = [-q] * (M + 1)
        f = [0] * (M + 1)
        for j in range(M + 1):
            f[j] = U[i][j] + tau * phi(A + j * h, tau * i)

        b[0] = 1
        c[0] = 0
        f[0] = ua(tau * (i + 1))

        a[M] = 0
        b[M] = 1
        f[M] = ub(tau * (i + 1))
        
        alpha[0] = -c[0] / b[0]
        beta[0] = f[0] / b[0]

        for j in range(1, M + 1):
            alpha[j] = -c[j] / (a[j] * alpha[j - 1] + b[j])
            beta[j] = (f[j] - a[j] * beta[j - 1]) / (a[j] * alpha[j - 1] + b[j])

        U[i + 1][M] = (f[M] - a[M] * beta[M - 1]) / (a[M] * alpha[M - 1] + b[M])

        for j in range(M - 1, -1, -1):
            U[i + 1][j] = alpha[j] * U[i + 1][j + 1] + beta[j]

    for i in range(M + 1):
        U[0][i] = psi(A + i * h)
        
    new_U = np.array([])

    for i in range(N + 1):
        for j in range(M + 1):
            new_U = np.append(new_U, U[i][j])
    #print('new_U[0] =', new_U)
    return new_U

def func(N1, M1, N2, M2, N3, M3, N4, M4):
    U1 = solve(N1, M1)
    U2 = solve(N2, M2)
    U3 = solve(N3, M3)
    U4 = solve(N4, M4)

    h1 = (B - A) / M1
    tau1 = T / N1
    h2 = (B - A) / M2
    tau2 = T / N2
    h3 = (B - A) / M3
    tau3 = T / N3
    h4 = (B - A) / M4
    tau4 = T / N4

    x1 = np.arange(A, B + h1, h1)
    t1 = np.arange(0, T + tau1, tau1)
    x2 = np.arange(A, B + h2, h2)
    t2 = np.arange(0, T + tau2, tau2)
    x3 = np.arange(A, B + h3, h3)
    t3 = np.arange(0, T + tau3, tau3)
    x4 = np.arange(A, B + h4, h4)
    t4 = np.arange(0, T + tau4, tau4)

    print(N1, len(U1))
    print(N2, len(U2))
    print(N3, len(U3))
    print(N4, len(U4))

    new_U1 = np.array(np.split(U1, N1 + 1))
    new_U2 = np.array(np.split(U2, N2 + 1))
    new_U3 = np.array(np.split(U3, N3 + 1))
    new_U4 = np.array(np.split(U4, N4 + 1))

    e1, f1 = np.meshgrid(x1, t1)
    e2, f2 = np.meshgrid(x2, t2)
    e3, f3 = np.meshgrid(x3, t3)
    e4, f4 = np.meshgrid(x4, t4)

    ax.plot_wireframe(e1, f1, new_U1, color = 'black')
    ax.plot_wireframe(e2, f2, new_U2, color = 'green')
    ax.plot_wireframe(e3, f3, new_U3, color = 'red')
    ax.plot_wireframe(e4, f4, new_U4, color = 'orange')

    plt.show()
func(40, 40, 120, 40, 40, 120, 120, 120)

