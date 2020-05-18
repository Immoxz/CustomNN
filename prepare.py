# making loss function
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from util import generate_data2d4c, generate_data2d2c, plot_data

data2c, labels2c = generate_data2d4c(1000, 3)
o_slo = {1: 0, -1: 1, -0.5: 2, 0.5: 3}

plot_data(data2c, labels2c, o_slo)
