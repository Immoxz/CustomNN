from node import *


class NodesLayer(object):
    def __init__(self, input_layer, output_layer, debug_mode=False):
        self.debug_mode = debug_mode
        self.num_input_nodes = input_layer
        self.num_output_nodes = output_layer
        self.layer = []
        self.__make_nodes()

    def __make_nodes(self):
        self.layer = [Node(self.num_input_nodes, self.debug_mode) for _ in range(self.num_output_nodes)]

    def get_nodes(self):
        return self.layer

    def forward(self, input_units):
        layer_prediction = []
        for node in self.layer:
            layer_prediction.append(node.forward(input_units))
        return layer_prediction

    def backward(self):
        for node in self.layer:
            node.backward()

    def update_params(self, step_size):
        for node in self.layer:
            node.update_params(step_size)
