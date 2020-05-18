# making loss function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util import generateData2D4C, generateData2D2C, plotData

data2c, labels2c = generateData2D4C(1000, 3)
o_slo = {1: 0, -1: 1, -0.5: 2, 0.5: 3}

plotData(data2c, labels2c, o_slo)
