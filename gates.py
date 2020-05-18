from units import *
import numpy as np

from util import min_max_scaling


class multiply_gate(object):
    def __init__(self):
        self.u0 = Unit
        self.inner = Unit
        self.utop = Unit

    def forward(self, u0, u1):
        # store pointers to input Units u0 and u1 and output unit utop
        self.u0 = u0
        self.inner = u1
        self.utop = Unit(u0.value * u1.value, 0.0)
        # print(self.utop.value)
        return self.utop

    def backward(self):
        # take the gradient in output unit and chain it with the
        # local gradients, which we derived for multiply gate before
        # then write those gradients to those Units.
        self.u0.grad += self.inner.value * self.utop.grad
        if self.inner.value != 0:
            self.inner.grad += self.u0.value * self.utop.grad
        else:
            self.inner.grad = 0


class add_gate(object):
    def __init__(self):
        self.u0 = Unit
        self.u1 = Unit
        self.utop = Unit

    def forward(self, u0, u1):
        # store pointers to input Units u0 and u1 and output unit utop
        self.u0 = u0
        self.u1 = u1
        self.utop = Unit(u0.value + u1.value, 0.0)
        return self.utop

    def backward(self):
        # add gate. derivative wrt both inputs is 1
        self.u0.grad += 1.0 * self.utop.grad
        self.u1.grad += 1.0 * self.utop.grad


class sigmoid_gate(object):
    def __init__(self):
        self.u0 = Unit
        self.utop = Unit

    def sig(self, x):
        return 1 / (1 + np.exp(np.float64(min_max_scaling(-x))))

    def forward(self, u0):
        # store pointers to input Units u0 and u1 and output unit utop
        self.u0 = u0
        self.utop = Unit(self.sig(self.u0.value), 0.0)
        return self.utop

    def backward(self):
        s = self.sig(self.u0.value)
        self.u0.grad += (s * (1 - s)) * self.utop.grad
