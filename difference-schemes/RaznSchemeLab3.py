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
m = 25
r = 1.2
h = (b - a) / m
tau = r * h
n = int(T / tau) + 1
t = np.arange(0, T + 2 * tau, tau)
X = np.arange(a, b + h * n + 2 * h, h)

U = []
U_level = np.array([])
new_U = np.array([])
Y_level = np.array([])
Y = []
new_Y = np.array([])

for i in range(m + n + 2):
    U_level = np.append(U_level, psi(X[i]))
U.append(U_level)

for i in range(m + 2):
    Y_level = np.append(Y_level, psi(X[i]))
Y.append(U_level)

for i in range(1, n + 2):
    U_level = np.array([])
    for k in range(0, m + n + 2 - i):
        U_level = np.append(U_level, (1 - c * r) * U[i - 1][k] + c * r * U[i - 1][k + 1] + tau * phi(X[k], t[i - 1]))
    U.append(U_level)

for i in range(1, n + 2):
    Y_level = np.array([])
    for j in range(0, m + 2):
        Y_level = np.append(Y_level, np.log(t[i] + 1) + t[i] * (np.log(t[i] + 1) -
                                1 - 3 * X[j] - 3 * c * t[i]) + 3 / 2 * c * (t[i] ** 2) + np.e ** (-(X[j] + c * t[i]) / 4))
    Y.append(Y_level)


for i in range(n + 1):
    for j in range(m + 1):
        new_U = np.append(new_U, U[i][j])
        new_Y = np.append(new_Y, Y[i][j])


X = np.arange(a, b + h, h)
t = np.arange(0, T + tau, tau)
print('new_U len =', len(new_U))
print('new_Y len =', len(new_Y))
print('n =', n)
print('m =', m)


new_U = np.array(np.split(new_U, n + 1)) 
new_Y = np.array(np.split(new_Y, n + 1))
e, f = np.meshgrid(X, t)
print('shape(e) =', e.shape)
print('shape(f) =', f.shape)
print('shape(new_U) =', new_U.shape)
print('shape(new_Y) =', new_Y.shape)

ax.plot_wireframe(e, f, new_U, color = 'black')
ax.plot_wireframe(e, f, new_Y, color = 'green')
plt.show()
