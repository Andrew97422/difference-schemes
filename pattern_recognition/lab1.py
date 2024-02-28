import numpy as np
import matplotlib.pyplot as plt


#  Функция для вычисления расстояния
def ro(X, Z):
    return np.sqrt(np.sum((X - Z) ** 2))


#  Разделяющая граница
def d(x, Z1, Z2):
    return (-(Z1[0] - Z2[0]) * x + (Z1[0] ** 2 + Z1[1] ** 2 - Z2[0] ** 2 - Z2[1] ** 2) / 2) / (Z1[1] - Z2[1])


W1 = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
W2 = np.array([[10, 0], [10, 1], [11, 0], [11, 1], [9, 5]])
X = [3, 5]

plt.scatter(W1[:, 0], W1[:, 1], c='b', s=10)
plt.scatter(W2[:, 0], W2[:, 1], c='r', s=10)

Z1 = W1.mean(axis=0)
Z2 = W2.mean(axis=0)
plt.scatter(Z1[0], Z1[1], c='b', s=75)
plt.scatter(Z2[0], Z2[1], c='r', s=75)
x = np.arange(0, 11, 0.01)
plt.scatter(x, d(x, Z1, Z2), c='black', s=5)

round = 1e-3

if ro(X, Z2) - ro(X, Z1) > round:
    plt.scatter(X[0], X[1], c='b', s=30)
elif ro(X, Z1) - ro(X, Z2) > round:
    plt.scatter(X[0], X[1], c='r', s=30)
else:
    plt.scatter(X[0], X[1], c='black', s=30)


plt.axis('equal')
plt.show()
