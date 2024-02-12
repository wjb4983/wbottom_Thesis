import numpy as np
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
import math

#vascularnet.py

# Function to initialize vascular weights
def initialize_vascular_weights(num_nodes):
    return np.random.rand(num_nodes, num_nodes)


def sigmoid(x):
    # Sigmoid activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    # Derivative of the sigmoid activation function
    return sigmoid(x) * (1 - sigmoid(x))

def energy_gradient(delta_s, weights, g_prime_1, h_f):
    # Calculate the energy gradient based on the provided formula
    return np.sum(delta_s * weights * g_prime_1(h_f))

def h_f(weights, inputs, biases):
    # Calculate h_j_f based on the provided formula
    return np.dot(weights, inputs) - biases

def f(e_j):
    # Calculate f(e_j) based on the provided conditions
    if 0 <= e_j <= 2:
        return 1 - e_j
    elif e_j > 2:
        return -1

def b_f(bias_factor, delta_j_s, g_prime_1_val, h_j_f):
    # Calculate b_j_f based on the provided formula
    return -bias_factor * delta_j_s * g_prime_1_val


def h_i_s(weights, inputs, biases):
    # Calculate h_i_s based on the provided formula
    return np.dot(weights, inputs) - biases

def b_i_s(bias_factor, delta_i_s, g_prime_1, h_i_s):
    # Calculate b_i_s based on the provided formula
    return -bias_factor * delta_i_s * g_prime_1(h_i_s)

def delta_s(e_i, g_prime_2, h_i_s):
    # Calculate delta_i_s based on the provided formula
    return e_i * g_prime_2 * h_i_s

def forward_pass_vascular_network(energy_source, vascular_weights, bias_factor=0.005):
    energy_levels = [np.zeros(energy_source.shape[0])]
    
    for i, weights in enumerate(vascular_weights):
        h_j_f = h_f(weights, energy_levels[-1], 0)  # Assuming 0 as biases for simplicity
        g_prime_1_val = sigmoid_derivative(h_j_f)
        delta_j_s = delta_s(energy_levels[-1], g_prime_1_val, h_j_f)
        b_j_f = b_f(bias_factor, delta_j_s, g_prime_1_val, h_j_f)
        
        # Update the bias for the source layer
        # Assuming the source layer is the first element of the weights list
        vascular_weights[0] -= b_j_f
        
        energy_levels.append(np.dot(weights, energy_levels[-1]))

    return energy_levels




# Replace gradient_of_energy_function(weights) with the actual calculation of the gradient

# Function to simulate the forward pass of the vascular network
def forward_pass_vascular_network_simpl(energy_source, vascular_weights):
    energy_levels = [np.zeros(energy_source.shape[0])]
    
    for weights in vascular_weights:
        energy_levels.append(np.dot(weights, energy_levels[-1]))

    return energy_levels

class ClassifierLayer(tf.keras.layers.Layer):
    def __init__(self, numhidden):
        global i
        super(ClassifierLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(numhidden, activation='linear', name=str(i))
        

    def build(self, input_shape):
        # input_dim = input_shape[-1]
        self.bias = self.add_weight("bias", shape=(1,), initializer="zeros", trainable=True)

    def call(self, inputs, energy_levels, vasc_done, vasc_out):
        if(vasc_done):
            self.bias = 0;
            self.bias = self.add_weight("bias", shape=(1,), initializer=vasc_out)
        
        dense_weights = self.dense.get_weights()
        
        # Get the weights of the dense layer
        weights = dense_weights[0]
        # Element-wise multiplication
        weighted_output = inputs * weights
        
        # Sum across input_size to get a single value for each neuron
        neuron_sums = tf.reduce_sum(weighted_output, axis=0)
        print(tf.size(neuron_sums).numpy())
        classifier_output = neuron_sums + energy_levels
        return classifier_output

# Function to simulate the forward pass of the neural network
def forward_pass_neural_network(input_data, weights, biases):
    hidden_layer_input = np.dot(input_data, weights) + biases
    hidden_layer_output = 1 / (1 + np.exp(-hidden_layer_input))
    return hidden_layer_output

# Function to train the neural network
def train_neural_network(X, y, numhidden):
    clf = MLPClassifier(hidden_layer_sizes=(numhidden,), max_iter=1000)
    clf.fit(X, y)
    return clf.coefs_[0].T, clf.intercepts_[0]  # Transpose weights

def simulate_neurovascular_network(X, y, numhidden, branching_factor=1):
    num_hidden_neurons = numhidden

    # Training neural network
    weights, biases = train_neural_network(X, y, numhidden)

    # Calculate the number of layers based on the tree structure


    if(branching_factor == 1):
        num_layers = 1
    else:
        num_layers = int(1 + math.ceil(math.log(numhidden) / math.log(branching_factor)))
        
    # Calculate the number of nodes in each layer
    nodes_per_layer = [1] + [branching_factor**i for i in range(1, num_layers)]

    # Create a list of tuples specifying the number of nodes in each layer
    layer_sizes = tuple(nodes_per_layer)
    # Initializing vascular weights
    num_nodes = num_hidden_neurons * branching_factor
    vascular_weights = initialize_vascular_weights(num_nodes)

    # Simulating forward pass of the vascular network
    energy_source = np.ones(num_nodes)  # Assuming the energy source is connected to the root node
    energy_levels = forward_pass_vascular_network(energy_source, vascular_weights)

    # Ensure that the number of hidden neurons in the neural network matches the size of input_data
    if num_hidden_neurons != energy_levels[-1].shape[0]:
        raise ValueError(f"Number of hidden neurons ({num_hidden_neurons}) in the neural network does not match the size of input_data ({energy_levels[-1].shape[0]}).")

    # Reshape input_data to match the expected shape of the neural network
    input_data = energy_levels[-1].reshape(1, -1)


    # Simulating forward pass of the neural network
    # hidden_layer_output = forward_pass_neural_network(input_data, weights, biases[0])  # Use biases[0]
    # print("Neural Network Output:", hidden_layer_output)
    # print(hidden_layer_output.size)
    
    # # Return the neural network output
    # return hidden_layer_output
    print("Neural Network Output:", energy_levels)


    # Return the neural network output
    return energy_levels

if __name__ == "__main__":
    # Assuming X and y are your training data and labels
    X = np.random.rand(100, 1)  # Replace with your actual data
    y = np.random.randint(2, size=100)  # Replace with your actual labels

    simulate_neurovascular_network(X, y, 10)
