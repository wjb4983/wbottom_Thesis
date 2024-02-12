# outdated - new version in MIFinalSequentiallyTrained











import numpy as np
import networkx as nx  # You can install networkx using: pip install networkx

class VascNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Create adjacency list
        self.adjacency_list = {}

        # Input layer to hidden layer
        self.add_connections(range(input_size), range(input_size, input_size + hidden_size))

        # Hidden layer to output layer
        self.add_connections(range(input_size, input_size + hidden_size), range(input_size + hidden_size, input_size + hidden_size + output_size))

    def add_connections(self, from_nodes, to_nodes):
        for from_node in from_nodes:
            self.adjacency_list[from_node] = []
            for to_node in to_nodes:
                weight = np.random.randn()  # Replace with your weight initialization method
                self.adjacency_list[from_node].append({'to': to_node, 'weight': weight})

    def forward_propagation(self, input_values):
        # Initialize output values
        output_values = {i: 0.0 for i in range(self.input_size + self.hidden_size + self.output_size)}

        # Set input values
        for i, value in enumerate(input_values):
            output_values[i] = value

        # Perform forward propagation
        for from_node in self.adjacency_list:
            for connection in self.adjacency_list[from_node]:
                to_node = connection['to']
                weight = connection['weight']
                output_values[to_node] += output_values[from_node] * weight

        return output_values

# Example usage
input_size = 3
hidden_size = 4
output_size = 2

nn = VascNetwork(input_size, hidden_size, output_size)

# Forward propagation
input_values = [1.0, 2.0, 3.0]
output_values = nn.forward_propagation(input_values)

print("Input values:", input_values)
print("Output values:", [output_values[i] for i in range(input_size + hidden_size, input_size + hidden_size + output_size)])
