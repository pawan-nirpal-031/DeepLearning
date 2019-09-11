import numpy as np
from matplotlib import pyplot as plt

# each data point is [x,y,z] x is length , y is width z is {0,1} 1 if red 0 if blue

data = [[3, 1.5, 1],
        [2, 1, 0],
        [4, 1.5, 1],
        [3, 1, 0],
        [3.5, 0.5, 1],
        [2, 0.5, 0],
        [5.5, 1, 1],
        [1, 1, 0]]

unknown_flower = [4.5, 1]


def node(x, y, w1, w2, b):
    z = x*w1 + y*w2 + b
    return sigmoid(z)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def cost(b):
    return (b-4)**2


def numCost(b):
    h = 0.0001
    return (cost(b+h)-cost(b))/h


def slope(b):
    return 2*(b-4)


def deriv_sigmoid(x):
    return sigmoid(x) * (1-sigmoid(x))


w1x = np.random.randn()

w2y = np.random.randn()

bias = np.random.randn()

# x1 = int(input("Enter patel length"))
# y1 = float(input("Enter patel width"))
# out = node(x1, y1, w1x, w2y, bias)
# print('chances of this flower being red close to (1) are {}'.format(out))
p =-10


for i in range(50):
    p = p - 0.1 * slope(p)
    print(p)


X = np.linspace(-20, 20, 100)

Y = deriv_sigmoid(X)

plt.plot(X, Y)

plt.xlabel = 'X axis'
plt.ylabel = 'Y axis'
plt.show()