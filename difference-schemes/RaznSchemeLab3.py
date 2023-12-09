import math
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(projection='3d')


def phi(x, t):
    return np.log(1 + t) - 3 * x


def psi(x):
    return np.e ** (-x / 4)


a = 0.2
b = 2.2
c = 0.4
T = 1.9
h = 0.1 # По x
M = int((b - a) / h)
r = 0.5
tau = r * h # По t
N = int((T - 0) / tau) + 1
print(N)
X = np.arange(a, b + (N + 1) * h, h)
# print(h)

t = np.arange(0, T, tau)
U = []
U_level = np.array([])
new_U = np.array([])
Y_level = np.array([])
Y = []
new_Y = np.array([])


# Точное решение
def exactSolution(Y):
    for i in range(1, N): # i идёт по t (по сути n)
        Y_level = np.array([])
        for j in range(0, M): # j идёт по x (по сути m)
            Y_level = np.append(Y_level, np.log(t[i] + 1) + t[i] * (np.log(t[i] + 1) -
                                1 - 3 * X[j] - 3 * c * t[i]) + 3 / 2 * c * (t[i] ** 2) + np.e ** (-(X[j] + c * t[i]) / 4))
        Y.append(Y_level)
    return Y


# Численное решение
def numericalSolution(U):
    for i in range(1, N): # i идёт по t (по сути n)
        print('i = ', i, 'N = ', N)
        U_level = np.array([])
        for j in range(0, M + N - i): # j идёт по x (по сути m)
            U_level = np.append(U_level, (1 - c * r) * U[i - 1][j] + c * r * U[i - 1][j + 1] + tau * phi(X[j], t[i - 1]))
        U.append(U_level)
    return U


# Нулевой уровень для численного решения
for i in range(0, M + N):
    U_level = np.append(U_level, psi(X[i]))
U.append(U_level)


# Нулевой уровень для точного решения
for i in range(0, M):
    Y_level = np.append(Y_level, psi(X[i]))
Y.append(U_level)


# Численное решение
U = numericalSolution(U)

# Точное решение
Y = exactSolution(Y)


# Допоперации
for i in range(0, N):
    for j in range(0, M):
        new_U = np.append(new_U, U[i][j])
        new_Y = np.append(new_Y, Y[i][j])
        
# Ограничения на X
X = np.arange(a, b, h)

# Приведение к нужному виду для построения графика
new_U = np.array(np.split(new_U, N)) 
new_Y = np.array(np.split(new_Y, N))
e, f = np.meshgrid(X, t)

ax.plot_wireframe(e, f, new_U, color = 'black')
ax.plot_wireframe(e, f, new_Y, color = 'green')
plt.show()
