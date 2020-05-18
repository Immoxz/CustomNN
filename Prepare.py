# making loss function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util import generateDate2D4C, generateDate2D2C, plotData

data2c, labels2c = generateDate2D4C(1000, 3)
o_slo = {1: 0, -1: 1, -0.5: 2, 0.5: 3}

plotData(data2c, labels2c, o_slo)
