from circuit import *


class SVM(object):
    def __init__(self):
        self.a = Unit(1.0, 0.0)
        self.b = Unit(-2.0, 0.0)
        self.c = Unit(2.0, 0.0)
        self.circuit = Circuit()

        self.unit_out = Unit

    def forward(self, x, y):  # assuming that x and y are Units
        self.unit_out = self.circuit.forward(x, y, self.a, self.b, self.c)
        return self.unit_out

    def backward(self, label):  # label is +1 or -1
        # reset pulls on a,b,c
        self.a.grad = 0.0
        self.b.grad = 0.0
        self.c.grad = 0.0

        # compute the pull base on what the circuit output was
        pull = 0.0
        if label == 1 and self.unit_out.value < 1:
            pull = 1  # the score was too low: pull up
        if label == -1 and self.unit_out.value > -1:
            pull = -1  # the score was too high for a positive example, pull down

        self.circuit.backward(pull)  # writes gradient into x,y,a,b,c

        # add regularization pull for parameters: towards zero and proportional to value
        self.a.grad += -self.a.value
        self.b.grad += -self.b.value

    def parametersUpdate(self):
        step_size = 0.01
        self.a.value += step_size * self.a.grad
        self.b.value += step_size * self.b.grad
        self.c.value += step_size * self.c.grad

    def learnFrom(self, x, y, label):
        self.forward(x, y)  # forward pass (set .value in all Units)
        self.backward(label)  # backward pass(set .grad in all units)
        self.parametersUpdate()  # parameters respond to tug
