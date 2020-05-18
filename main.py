import os

from units import Unit
from util import generateDate2D4C

net = loadParametes(os.path.join('C:\\Users\\piotr.lejman\\Desktop\\NNparameters', 'trained_net_I2_h9_o4_100PERC.txt'))
ok = 0
checkData, checkLabels = generateDate2D4C(1000)
for iter2 in range(len(checkData)):
    unitX = Unit(checkData[iter2][0], 0.0)
    unitY = Unit(checkData[iter2][1], 0.0)
    label = checkLabels[iter2]
    inputUnits = [unitX, unitY]

    out_unit = net.forward(inputUnits)

    if True and iter2 % 25 == 0:
        [print('score' + str(i), label, round(out.value, 7)) for i, out in enumerate(out_unit)]
        print(' ')
    o_slo = {1: 0, -1: 1, -0.5: 2, 0.5: 3}

    # print(label, o_slo[label], out_unit[o_slo[label]].value)
    plus = 0
    for i, out in enumerate(out_unit):
        judge = o_slo[label]
        if i == o_slo[label]:
            if out.value > .9:
                plus += 1
        else:
            if out.value < .1:
                plus += 1
    if plus == 4:
        ok += 1

wynik = ok / len(checkData)
print(wynik)
