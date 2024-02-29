import numpy as np
import matplotlib.pyplot as plt


#  Разделяющая граница
def d(x, Z1, Z2):
    return (-(Z1[0] - Z2[0]) * x + (Z1[0] ** 2 + Z1[1] ** 2 - Z2[0] ** 2 - Z2[1] ** 2) / 2) / (Z1[1] - Z2[1])


#  Критерий классификации
def d_func(X, Z1, Z2):
    return (Z1[0] -Z2[0]) * X[0] + (Z1[1] - Z2[1]) * X[1] + (Z2[0] ** 2 + Z2[1] ** 2 - Z1[0] ** 2 - Z1[1] ** 2) / 2


W1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W2 = np.array([[10, 0], [10, 1], [11, 0], [11, 1], [9, 5]])
X = [3, 3]

plt.scatter(W1[:, 0], W1[:, 1], c='b', s=10)
plt.scatter(W2[:, 0], W2[:, 1], c='r', s=10)

Z1 = W1.mean(axis=0)
Z2 = W2.mean(axis=0)
plt.scatter(Z1[0], Z1[1], c='b', s=75)
plt.scatter(Z2[0], Z2[1], c='r', s=75)
x = np.arange(0, 11, 0.01)
plt.scatter(x, d(x, Z1, Z2), c='black', s=5)
y = np.arange(0, 11, 0.01)
for i in range(len(x)):
    y[i] = d(x[i], Z1, Z2)

round = 1e-3

if d_func(X, Z1, Z2) > 0:
    plt.scatter(X[0], X[1], c='b', s=30)
else:
    plt.scatter(X[0], X[1], c='r', s=30)

plt.xlim(-1, 12)
plt.ylim(-1, 6)
#plt.axis('equal')
plt.show()
