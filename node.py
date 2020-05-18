from util import *
from gates import *


class Node(object):
    def __init__(self, in_nodes, debug_mode=False):
        if debug_mode:
            print('Number of input nodes', in_nodes)
        self.debug_mode = debug_mode
        self.mul_gates = [multiplyGate() for m in range(in_nodes)]
        self.add_gates = [addGate() for a in range(in_nodes)]
        # self.sigg0 = sigmoidGate()
        self.mul_units = []
        self.add_units = []
        self.inner_units = [Unit(random.uniform(-1.0, 1.0), random.random() - 0.5) for iu in range(in_nodes)]
        self.native_unit = Unit(random.uniform(-1.0, 1.0), random.random() - .5)
        self.out_unit = Unit
        self.nodes_inside = len(self.inner_units)

    def forward(self, input_units_table):

        self.mul_units = []
        # multiplay gates
        self.mul_units = [mulGate.forward(input_units_table[index], self.inner_units[index]) for index, mulGate in
                          enumerate(self.mul_gates)]
        self.add_units = []
        # additional cascade gates
        for aGi, add_gate in enumerate(self.add_gates):
            if aGi == 0:
                self.add_units = [add_gate.forward(self.mul_units[0], self.mul_units[1])]
            elif len(self.mul_units) - 1 > aGi > 0:
                self.add_units.append(add_gate.forward(self.add_units[-1], self.mul_units[aGi + 1]))
            if len(self.mul_units) - 1 == aGi:
                self.add_units.append(add_gate.forward(self.add_units[-1], self.native_unit))

        self.out_unit = self.add_units[-1]
        # sigmoid gate
        # self.out_unit = self.sigg0.forward(self.add_units[-1])

        if self.debug_mode:
            [print('input ', x.value, x.grad) for x in input_units_table]
            [print('inner ', x.value, x.grad) for x in self.inner_units]
            print('nativeUnit ', self.native_unit.value, self.native_unit.grad)
            print(' ')

            [print('multi ', x.value, x.grad) for x in self.mul_units]
            print(' ')

            [print('add ', x.value, x.grad) for x in self.add_units]
            print(' ')

            print('sig ', self.out_unit.value, self.out_unit.grad)
            print('-------------------')

        return self.out_unit

    def backward(self):
        # self.sigg0.backward()
        for add_gate in reversed(self.add_gates):
            add_gate.backward()
        for mul_gate in reversed(self.mul_gates):
            mul_gate.backward()

        if self.debug_mode:
            print('back sig ', self.out_unit.value, self.out_unit.grad)
            print(' ')
            [print('back add ', x.value, x.grad) for x in self.add_units]
            print(' ')
            [print('back multi ', x.value, x.grad) for x in self.mul_units]
            print(' ')
            print('-----------------------------------------------')

    def updateParams(self, step_size):
        for inner_unit in self.inner_units:
            inner_unit.value += step_size * inner_unit.grad
        self.native_unit.value += step_size * self.native_unit.grad

    def getSize(self):
        return self.nodes_inside

    def getOutputUnit(self):
        return self.out_unit

    def getInnerUnits(self):
        return self.inner_units
