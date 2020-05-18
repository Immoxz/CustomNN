from layer import *


class Network(object):
    def __init__(self, input_num, output_num, hidden_tab=None):
        if hidden_tab is None:
            hidden_tab = []
        self.are_hidden_layers = True
        self.output_num_nodes = output_num
        self.input_num_nodes = input_num
        if len(hidden_tab) == 0:
            self.are_hidden_layers = False
        elif all(layer == 0 for layer in hidden_tab):
            self.are_hidden_layers = False
        if self.are_hidden_layers:
            self.table_hidden_layer = [x for x in hidden_tab if x != 0]

        self.network = []
        self.__makeNetwork()

    # building network depends on how
    def __makeNetwork(self):
        if not self.are_hidden_layers:
            self.__makeDefault(self.input_num_nodes, self.output_num_nodes)
        else:
            self.__makeHidden(self.input_num_nodes, self.output_num_nodes, self.table_hidden_layer)

    def __makeDefault(self, inputNumNodes, outputNumNodes):
        self.network = [NodesLayer(inputNumNodes, outputNumNodes)]

    def __makeHidden(self, num_input_nodes, num_output_nodes, table_hidden_nodes):
        # creating layer of nodes input
        input_layer = [NodesLayer(num_input_nodes, table_hidden_nodes[0])]
        # creating hidden layers
        hidden_layers = []
        if len(table_hidden_nodes) != 1:
            for index, layerNum in enumerate(table_hidden_nodes):
                if index > 0:
                    hidden_layers.append(NodesLayer(table_hidden_nodes[index - 1], layerNum))
        # last output layer
        output_layer = [NodesLayer(table_hidden_nodes[-1], num_output_nodes)]
        # saving all to global network
        self.network = input_layer + hidden_layers + output_layer

    def forward(self, input_units):
        network_predictions = []
        for index, layer in enumerate(self.network):
            if index == 0:
                network_predictions.append(layer.forward(input_units))
            else:
                network_predictions.append(layer.forward(network_predictions[-1]))
        return network_predictions[-1]

    def backward(self):
        for layer in reversed(self.network):
            layer.backward()

    def update_parameters(self, step_size):
        for layer in self.network:
            layer.update_params(step_size)
