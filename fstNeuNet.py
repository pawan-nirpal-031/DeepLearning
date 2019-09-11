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


# out = node(x1, y1, w1x, w2y, bias)
# print('chances of this flower being red close to (1) are {}'.format(out))


X = np.linspace(-6, 6, 100)

Y = deriv_sigmoid(X)
# plt.title('Sigmoid function in Neural Network Training')
# plt.plot(X, sigmoid(X), label='Sigmoid(x)', c='r')
# plt.plot(X, deriv_sigmoid(X), label='Derivative_of_Sigmoid(x)', c='b')
# plt.legend()
# plt.xlabel("X axis ")
# plt.ylabel("Y axis ")
# plt.show()



for i in range(len(data)): # scatter plot of data
    plt.axis([0, 6, 0, 6])
    point = data[i] # point is actually this [length, width, type], where type is 0 or 1
    color = 'r'
    if point[2] == 0:
        color = 'b'
    plt.scatter(point[0], point[1], c=color)


#plt.grid()
#plt.show()
learning_rate = 0.1


for i in range(10000):  # training loop
    ri = np.random.randint(int(len(data)))
    point = data[ri]
    z = point[0]*w1x + point[1]*w2y + bias
    pred = sigmoid(z)
    target = point[2]
    cost = np.square(pred - target)
    if i%1000 == 0:
        print('Cost now : ', cost)
    deriv_cost = 2*(pred - target)
    deriv_predWz = deriv_sigmoid(z)
    dwz_w1 = point[0]
    dwz_w2 = point[1]
    dz_b = 1
    dcost_dz = deriv_cost*deriv_predWz
    dcost_dw1 = dcost_dz*dwz_w1
    dcost_dw2 = dcost_dz*dwz_w2
    dcost_db = dcost_dz*dz_b
    w1x = w1x - learning_rate*dcost_dw1
    w2y = w2y - learning_rate*dcost_dw2
    bias = bias - learning_rate*dcost_db
