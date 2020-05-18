from gates import *


class Circuit(object):
    def __init__(self):
        self.mulg0 = multiply_gate()
        self.mulg1 = multiply_gate()
        self.addg0 = add_gate()
        self.addg1 = add_gate()
        self.ax = Unit
        self.by = Unit
        self.axpby = Unit
        self.axpbyc = Unit

    def forward(self, x, y, a, b, c):
        self.ax = self.mulg0.forward(u0=a, u1=x)
        self.by = self.mulg1.forward(u0=b, u1=y)
        self.axpby = self.addg0.forward(u0=self.ax, u1=self.by)
        self.axpbyc = self.addg0.forward(u0=self.axpby, u1=c)
        return self.axpbyc

    def backward(self, gradient_top):
        self.axpbyc.grad = gradient_top  # takes pull from above
        self.addg1.backward()  # sets gradient in axpby and c
        self.addg0.backward()  # sets gradient in ax and by
        self.mulg1.backward()  # sets gradient in b and y
        self.mulg0.backward()  # sets gradient in a and x
