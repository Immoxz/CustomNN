import random

from network import *
from util import generate_data2d4c, plot_data

data2c, labels2c = generate_data2d4c(400)
# data2c, labels2c = loadParametes(os.path.join('C:\\Users\\piotr.lejman\\Desktop\\NNparameters', 'data2c.txt'))
input_nodes = len(data2c[0])
hidden_nodes1 = [11,7,4]
output_nodes = 4
max = len(data2c) * 800
debug = False
labels = [1, -1, -0.5, 0.5]
wynik = 0
def_bot = bottom_good = 0.3
def_top = top_good = 0.4

pull_up = 1.0
pull_down = -round(random.uniform(bottom_good, top_good), 5)
print(pull_up, pull_down)

o_slo = {1: 0, -1: 1, -0.5: 2, 0.5: 3}

net = Network(input_nodes, output_nodes, hidden_nodes1)
for iter in range(max):
    # input nodes
    choice = int(len(data2c) * random.random())
    label = labels2c[choice]

    # -----forward-----
    out_unit = net.forward([Unit(data2c[choice][0], 0.0), Unit(data2c[choice][1], 0.0)])
    # -----end forward-----

    pulls = calc_pull(out_unit, labels, label, pull_up, pull_down)
    # print(lossFunction(out_unit, label, o_slo[label]))
    # --------backprop---------
    for i in range(len(pulls)):
        out_unit[i].grad = pulls[i]

    net.backward()
    # --------end backprop---------

    # parameters update
    if (iter) == 0:
        step_size = 0.001
    else:
        step_size += -0.00001 / iter

    net.update_parameters(step_size)

    # debug print
    print_perc(iter, max)

ok = 0.0

checkData, checkLabels = generate_data2d4c(1000)
for iter2 in range(len(checkData)):
    unitX = Unit(checkData[iter2][0], 0.0)
    unitY = Unit(checkData[iter2][1], 0.0)
    label = checkLabels[iter2]
    inputUnits = [unitX, unitY]

    out_unit = net.forward(inputUnits)

    if True and iter2 % 250 == 0:
        [print('score' + str(i), label, round(out.value, 7)) for i, out in enumerate(out_unit)]
        print(' ')

    plus = 0
    for i, out in enumerate(out_unit):
        judge = o_slo[label]
        if i == o_slo[label]:
            if out.value > .99:
                plus += 1
        else:
            if out.value < .1:
                plus += 1
    if plus == 4:
        ok += 1

wynik = ok / len(checkData)
print(ok / len(checkData), bottom_good, top_good)

# plotData(data2c, labels2c, o_slo)
