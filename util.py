import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
# from yaml import load, dump

from units import Unit


# try:
#     from yaml import CLoader as Loader, CDumper as Dumper
# except ImportError:
#     from yaml import Loader, Dumper


def makeLogUnderMax(y, max):
    if y == 0:
        ly = 0
    else:
        ly = np.math.log(y, 2)
    return ly / np.math.log(max, 2)


def postProcessGameCube(X):
    return np.array([makeLogUnderMax(y, max(X)) for y in X])


#
# def saveParameters(*args):
#     path = args[0]
#     fileName = args[1]
#     params = dump(args[2], Dumper=Dumper)
#     f = open(os.path.join(path, fileName), 'w')
#     f.write(params)
#     f.close()
#
#
# def loadParametes(filePath):
#     f = open(filePath)
#     data = load(f.read(), Loader=Loader)
#     if len(data.keys()) != 1:
#         return_data = [data[key] for key in data.keys()]
#     else:
#         key = list(data.keys())[0]
#         return_data = data[key]
#     f.close()
#     return return_data


def shuffleDateWithClasses(data, classses):
    if len(data) == len(classses) and len(data) == 1:
        return data, classses
    elif len(data) == len(classses) and len(data) != 0:
        for i in range(100000):
            direction = random.random() - 0.5
            pos = int(random.random() * len(data) - 1)
            moveFor = int(random.randrange(1, len(data)) / 2)
            if moveFor + pos >= len(data) - 1:
                moveFor = -int(random.randrange(1, len(data)) / 2)

            if direction < 0:
                data[pos + moveFor], data[pos] = data[pos], data[pos + moveFor]
                classses[pos + moveFor], classses[pos] = classses[pos], classses[pos + moveFor]
            else:
                data[pos], data[pos + moveFor] = data[pos + moveFor], data[pos]
                classses[pos], classses[pos + moveFor] = classses[pos + moveFor], classses[pos]
        return data, classses
    else:
        raise ValueError('Length of data and classes are different.')


def generateDate2D4C(num, max_random=3):
    date = []
    classes = []
    for i in range(num):
        if i < int(1 * num / 4):
            date.append([round(random.uniform(0, max_random), 1), round(random.uniform(0, max_random), 1)])
            classes.append(-1.0)
        elif i < int(2 * num / 4):
            date.append([round(random.uniform(0, max_random), 1), round(random.uniform(-max_random, 0), 1)])
            classes.append(-0.5)
        elif i < int(3 * num / 4):
            date.append([round(random.uniform(-max_random, 0), 1), round(random.uniform(0, max_random), 1)])
            classes.append(0.5)
        else:
            date.append([round(random.uniform(-max_random, 0), 1), round(random.uniform(-max_random, 0), 1)])
            classes.append(1.0)
    return shuffleDateWithClasses(date, classes)


def generateDate2D2C(num, max_random=3):
    date = []
    classes = []
    for i in range(num):
        if i < num / 2:
            date.append([round(random.uniform(0, max_random), 1), round(random.uniform(0, max_random), 1)])
            classes.append(1)
        else:
            date.append([round(random.uniform(-max_random, 0), 1), round(random.uniform(-max_random, 0), 1)])
            classes.append(-1)
    return shuffleDateWithClasses(date, classes)


def transformToUtils(inputVector):
    return [Unit(x, 0.0) for x in inputVector]


def minMaxScaling(x, min=-10, max=10):
    return (x - min) / (max - min)


def storeAndAppendResults(path, result):
    data = ''
    if os.path.exists(path):
        k = open(path, 'r+')
        data = k.read()
        k.close()
    f = open(path, 'w')
    data += result
    f.write(data)
    f.close()


def calcPull(units_out, labels, label, pull_up, pull_down, limit=0.99):
    pulls = [pull_down for p in range(len(units_out))]
    for i in range(len(units_out)):
        if labels[i] == label and units_out[i].value <= limit:
            pulls[i] = pull_up
    return pulls


def printPerc(iter, max):
    if (iter % int((max / 10)) == 0 or iter + 1 == max) and True:
        bar = ['  ' for i in range(10)]
        for i in range(int((iter + 1) * 10 / max)):
            bar[i] = '=='
        print(''.join(bar), int((iter + 1) * 100 / max), '%')


def lossFunction(x, y, W):
    scores = W.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss = np.sum(margins)
    return loss


def plotData(data, labels, slo):
    r1 = np.asarray([x[0] for x in data])
    r2 = np.asarray([x[1] for x in data])
    l1 = np.asarray([slo[y] for y in labels])
    colors = ['red', 'green', 'blue', 'purple']
    plt.ylim([-3.1, 3.1])
    plt.xlim([-3.1, 3.1])
    plt.grid()
    plt.scatter(r1, r2, c=l1, cmap=matplotlib.colors.ListedColormap(colors))
    plt.show()
