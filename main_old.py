import math
import random

from NEAT_scratches.gates import *
from NEAT_scratches.units import *
from NEAT_scratches.circuit import *
from NEAT_scratches.classificators import *

# #
# a = Unit(1.0, 0.0)
# b = Unit(2.0, 0.0)
# c = Unit(-3.0, 0.0)
# x = Unit(-1.0, 0.0)
# y = Unit(3.0, 0.0)
#
# # create the gates
# mulg0 = multiplyGate()
# mulg1 = multiplyGate()
# addg0 = addGate()
# addg1 = addGate()
# sg0 = sigmoidGate()
#
#
# def forwardCircuitFast(a, b, c, x, y):
#     return 1 / (1 + math.exp(-(a * x + b * y + c)))
#
#
# def mathematicalCheck(a, b, c, x, y):
#     a = a.value
#     b = b.value
#     c = c.value
#     x = x.value
#     y = y.value
#     h = 0.0001
#     a_grad = (forwardCircuitFast(a + h, b, c, x, y) - forwardCircuitFast(a, b, c, x, y)) / h
#     b_grad = (forwardCircuitFast(a, b + h, c, x, y) - forwardCircuitFast(a, b, c, x, y)) / h
#     c_grad = (forwardCircuitFast(a, b, c + h, x, y) - forwardCircuitFast(a, b, c, x, y)) / h
#     x_grad = (forwardCircuitFast(a, b, c, x + h, y) - forwardCircuitFast(a, b, c, x, y)) / h
#     y_grad = (forwardCircuitFast(a, b, c, x, y + h) - forwardCircuitFast(a, b, c, x, y)) / h
#     print(a_grad, b_grad, c_grad, x_grad, y_grad)
#
#
# def forwardNeuron():
#     ax = mulg0.forward(u0=a, u1=x)
#     by = mulg1.forward(u0=b, u1=y)
#     axpby = addg0.forward(u0=ax, u1=by)
#     axpbypc = addg1.forward(u0=axpby, u1=c)
#     sg0.forward(u0=axpbypc)
#
#
# s = sg0.getUnitUtop()
# s.grad = 1.0
# sg0.setUnitUtop(s)
# i = 1
# while i != 120:
#     forwardNeuron()
#     s = sg0.getUnitUtop()
#     s.grad = 1.0
#     sg0.setUnitUtop(s)
#     sg0.backward()  # writes gradient into axpbypc
#     addg1.backward()  # writes gradients into axpby and c
#     addg0.backward()  # writes gradients into ax and by
#     mulg1.backward()  # writes gradients into b and y
#     mulg0.backward()  # writes gradients into a and x
#
#     step_size = 0.01
#     a.value += step_size * a.grad  # a.grad is -0.105
#     b.value += step_size * b.grad  # b.grad is 0.315
#     c.value += step_size * c.grad  # c.grad is 0.105
#     x.value += step_size * x.grad  # x.grad is 0.105
#     y.value += step_size * y.grad  # y.grad is 0.210
#
#     forwardNeuron()
#
#     i += 1
#
# print(a.grad, b.grad, c.grad, x.grad, y.grad)
# print(a.value, b.value, c.value, x.value, y.value)
#
# mathematicalCheck(a, b, c, x, y)

# ----------------------------

# SVN class
data = [[1.2, 0.7], [-0.3, -0.5], [3.0, 0.1], [-0.1, -1.0], [-1.0, 1.1], [2.1, -3], [2.0, 0.5]]
labels = [1, -1, 1, -1, -1, 1, 1]

svm = SVM()


#  a function that computes the classification accuracy
def evalTrainingAccuracy():
    num_correct = 0
    for i in range(len(data)):
        x = Unit(data[i][0], 0.0)
        y = Unit(data[i][1], 0.0)
        true_label = labels[i]

        if svm.forward(x, y).value > 0:
            predicted_label = 1
        else:
            predicted_label = -1

        if predicted_label == true_label:
            num_correct += 1

    return num_correct / len(data)


def cost(X, y, w, alpha):
    total_cost = 0.0  # L in SVM los function
    N = int(len(X))

    for i in range(N):
        # loop over all data points and compute their score
        xi = X[i]
        score = w[0] * xi[0] + w[1] * xi[1] + w[2]

        # accumulateing cost basen on how campatible the score is with the label
        yi = y[i]
        costi = max(0, -yi * score + 1)
        # print('example ' + str(i) + ': xi = (' + str(xi) + ') and label = ' + str(yi))
        # print('  score computed to be ' + str(round(score, 3)))
        # print('  => cost computed to be ' + str(round(costi, 3)))
        total_cost += costi

    reg_cost = alpha * (w[0] * w[0] + w[1] * w[1])
    print('regularization cost for current model is ' + str(round(reg_cost, 5)))
    total_cost += reg_cost

    print('total cost is ' + str(round(total_cost, 3)))
    return total_cost


input_nodes = 2
hidden_nodes = 5
output_nodes = 1

# learning loop
# for iter in range(400):
#     i = int(math.floor(random.random() * len(data)))
#     x = Unit(data[i][0], 0.0)
#     y = Unit(data[i][1], 0.0)
#     label = labels[i]
#     svm.learnFrom(x, y, label)
#
#     if iter % 25 == 0:
#         print('training accuracy at iter ' + str(iter) + ': ' + str(evalTrainingAccuracy()))

# ----------------------------

# a = 1
# b = -2
# c = -1  # initial parameters
# for iter in range(400):  # pick a random data point
#     i = int(math.floor(random.random() * len(data)))
#     x = data[i][0]
#     y = data[i][1]
#     label = labels[i]
#
#     # compute pull
#     score = a * x + b * y + c
#     pull = 0.0
#     if label == 1 and score < 1:
#         pull = 1;
#     if label == -1 and score > -1:
#         pull = -1;
#
#     # compute gradient and update parameters
#     step_size = 0.01
#     a += step_size * (x * pull - a)  # -a is from the regularization
#     b += step_size * (y * pull - b)  # -b is from the regularization
#     c += step_size * (1 * pull)
#     if iter % 25 == 0:
#         print('training accuracy at iter ' + str(iter) + ': ' + str(evalTrainingAccuracy()))

# ----------------------------

# initial parameters
# a
a1 = random.random() - 0.5
a2 = random.random() - 0.5
a3 = random.random() - 0.5
a4 = random.random() - 0.5
a5 = random.random() - 0.5
# b
b1 = random.random() - 0.5
b2 = random.random() - 0.5
b3 = random.random() - 0.5
b4 = random.random() - 0.5
b5 = random.random() - 0.5
# c
c1 = random.random() - 0.5
c2 = random.random() - 0.5
c3 = random.random() - 0.5
c4 = random.random() - 0.5
c5 = random.random() - 0.5
# d
d1 = random.random() - 0.5
d2 = random.random() - 0.5
d3 = random.random() - 0.5
d4 = random.random() - 0.5
d5 = random.random() - 0.5
# e
e5 = random.random() - 0.5

a = []

for iter in range(8000):  # pick a random data point
    i = int(math.floor(random.random() * len(data)))
    x = data[i][0]
    y = data[i][1]
    label = labels[i]

    # compute forward pass
    n1 = max(0, a1 * x + b1 * y + c1)
    n2 = max(0, a2 * x + b2 * y + c2)
    n3 = max(0, a3 * x + b3 * y + c3)
    n4 = max(0, a4 * x + b4 * y + c4)
    score = a5 * n1 + b5 * n2 + c5 * n3 + d5 * n4 + 1.0 * e5

    # compute pull
    pull = 0.0
    if label == 1 and score < 1:
        pull = 1
    if label == -1 and score > -1:

        pull = -1

    # now compute backward pass to all parameters of the model

    # backprop through the last "score" neuron
    dscore = pull

    da5 = n1 * dscore
    dn1 = a5 * dscore

    db5 = n2 * dscore
    dn2 = b5 * dscore

    dc5 = n3 * dscore
    dn3 = c5 * dscore

    dd5 = n4 * dscore
    dn4 = d5 * dscore

    de5 = 1.0 * dscore

    # backprop the ReLU non-linearities, in place
    # i.e. just set gradients to zero if the neurons did not "fire"
    if n4 == 0:
        dn4 = 0
    if n3 == 0:
        dn3 = 0
    if n2 == 0:
        dn2 = 0
    if n1 == 0:
        dn1 = 0

    # backprop to parameters of neuron 1
    da1 = x * dn1
    db1 = y * dn1
    dc1 = 1.0 * dn1

    # backprop to parameters of neuron 2
    da2 = x * dn2
    db2 = y * dn2
    dc2 = 1.0 * dn2

    # backprop to parameters of neuron 3
    da3 = x * dn3
    db3 = y * dn3
    dc3 = 1.0 * dn3

    # backprop to parameters of neuron 4
    da4 = x * dn4
    db4 = y * dn4
    dc4 = 1.0 * dn4

    # End of backprop!
    # note we could have also backproped into x,y
    # but we do not need these gradients. We only use the gradients
    # on our parameters in the parameter update, and we discard x,y

    # add the pulls from the regularization, tugging all multiplicative
    # parameters (i.e. not the biases) downward, proportional to their value
    da1 += -a1
    da2 += -a2
    da3 += -a3
    da4 += -a4

    db1 += -b1
    db2 += -b2
    db3 += -b3
    db4 += -b4

    da5 += -a5
    db5 += -b5
    dc5 += -c5
    dd5 += -d5

    # parameters update
    # step_size = 0.01
    if iter == 0:
        step_size = 0.01
    else:
        step_size += - (0.0005 / iter)
    a1 += step_size * da1
    b1 += step_size * db1
    c1 += step_size * dc1

    a2 += step_size * da2
    b2 += step_size * db2
    c2 += step_size * dc2

    a3 += step_size * da3
    b3 += step_size * db3
    c3 += step_size * dc3

    a4 += step_size * da4
    b4 += step_size * db4
    c4 += step_size * dc4

    a5 += step_size * da5
    b5 += step_size * db5
    c5 += step_size * dc5
    d5 += step_size * dd5
    e5 += step_size * de5
    if iter % 200 == 0:
        print('score ', round(score, 4))
        print('step ', round(step_size, 5))
        # if iter % 1000:
        #     cost(data, labels, [a1, b1, c1], 0.1)
        #     cost(data, labels, [a2, b2, c2], 0.1)
        #     cost(data, labels, [a3, b3, c3], 0.1)
    if score > 1:
        print('score ', round(score, 4))
        print('step ', round(step_size, 5))
        cost(data, labels, [a1, b1, c1], 0.1)
        cost(data, labels, [a2, b2, c2], 0.1)
        cost(data, labels, [a3, b3, c3], 0.1)
        break

# ----------------------------

# 2-D Support Vector Machine


X = [[1.2, 0.7], [-0.3, 0.5], [3, 2.5]]  # array of 2d data
y = [1, -1, 1]  # array of labels classes)
w = [0.1, 0.2, 0.3]  # example of random numbers
alpha = 0.1  # regularization strength


def cost(X, y, w):
    total_cost = 0.0  # L in SVM los function
    N = int(len(X))

    for i in range(N):
        # loop over all data points and compute their score
        xi = X[i]
        score = w[0] * xi[0] + w[1] * xi[1] + w[2]

        # accumulateing cost basen on how campatible the score is with the label
        yi = y[i]
        costi = max(0, -yi * score + 1)
        print('example ' + str(i) + ': xi = (' + str(xi) + ') and label = ' + str(yi))
        print('  score computed to be ' + str(round(score, 3)))
        print('  => cost computed to be ' + str(round(costi, 3)))
        total_cost += costi

    reg_cost = alpha * (w[0] * w[0] + w[1] * w[1])
    print('regularization cost for current model is ' + str(round(reg_cost, 3)))
    total_cost += reg_cost

    print('total cost is ' + str(total_cost))
    return total_cost

# cost(X, y, w)
