import tensorflow as tf
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Initializing vascular weights for a tree network
def initialize_vascular_weights(branching_factor, num_hidden):
    vascular_weights = []
    for i in range(num_hidden):
        connections = min(branching_factor, num_hidden - i)
        weights = tf.Variable(tf.random.normal((connections,)))
        vascular_weights.append(weights)
    return vascular_weights
# Custom layer for the tree structure
import math

i = 1

class TreeLayer(tf.keras.layers.Layer):
    def __init__(self, num_neurons, branching_factor, bias_factor):
        super(TreeLayer, self).__init__()
        self.num_neurons = num_neurons
        self.branching_factor = branching_factor
        self.bias_factor = bias_factor
        self.level = 1  # The level of the root node is set to 1 initially

        # Calculate the number of children for the current level
        self.num_children = math.ceil(self.num_neurons / (self.branching_factor**(self.level + 1)))
        self.num_children_with_k_branches = self.branching_factor if self.num_children >= self.branching_factor else self.num_children
        self.num_children_with_less_than_k_branches = self.num_children - self.num_children_with_k_branches

        # Ensure bias_factor is an integer (e.g., by rounding or converting)
        bias_factor_int = int(round(bias_factor))
        
        # Define the bias variable
        self.bias = self.add_weight("bias", shape=(bias_factor_int,), initializer="zeros")

        # Initialize num_nodes attribute
        self.num_nodes = 0

    def build(self, input_shape):
        pass

    def call(self, inputs):
        energy_levels = []

        # Expand the input to have a second dimension
        inputs = tf.expand_dims(inputs, axis=0)

        for i in range(self.num_children_with_k_branches):
            # Calculate the number of neurons for the child layer
            num_neurons_child = self.branching_factor

            # Create a dense layer for each child
            dense_layer = tf.keras.layers.Dense(units=num_neurons_child, activation="linear")
            energy_source_child = dense_layer(inputs)

            # Update the shape of self.bias
            self.bias = self.add_weight("bias", shape=energy_source_child.shape[1:], initializer="zeros")

            # Calculate energy gradient
            delta_j_s = energy_source_child * tf.nn.sigmoid(energy_source_child) * (1 - tf.nn.sigmoid(energy_source_child))
            b_j_f = -self.bias_factor * delta_j_s * tf.nn.sigmoid(energy_source_child) * (1 - tf.nn.sigmoid(energy_source_child))
            self.bias.assign_add(tf.squeeze(b_j_f, axis=0))  # Remove the extra dimension

            # Update energy source
            energy_source_child = energy_source_child - b_j_f

            energy_levels.append(tf.squeeze(energy_source_child, axis=0))  # Remove the extra dimension

        # Update num_nodes attribute
        self.num_nodes = sum([num_neurons_child for i in range(self.num_children_with_k_branches)])

        return energy_levels

class ClassifierLayer(tf.keras.layers.Layer):
    def __init__(self, numhidden):
        global i
        super(ClassifierLayer, self).__init__()
        self.dense = tf.keras.layers.Dense(numhidden, activation='linear', name=str(i))
        

    def build(self, input_shape):
        super(ClassifierLayer, self).build(input_shape)
        input_dim = input_shape[-1]
        # self.bias = self.add_weight("bias", shape=(input_dim,), initializer="zeros", trainable=False)

    def call(self, inputs, energy_levels):
        # Check if the dense layer has weights
        dense_weights = self.dense.get_weights()
        if not dense_weights:
            raise ValueError("Dense layer has no weights.")
        
        # Get the weights of the dense layer
        weights = dense_weights[0]
        # Element-wise multiplication
        weighted_output = inputs * weights
        
        # Sum across input_size to get a single value for each neuron
        neuron_sums = tf.reduce_sum(weighted_output, axis=0)
        print(tf.size(neuron_sums).numpy())
        classifier_output = neuron_sums + energy_levels
        return classifier_output

def forward_pass_neural_network(input_data, weights, biases):
    # Assuming input_data has shape (batch_size, input_size)
    hidden_layer_input = tf.matmul(input_data, weights) + biases
    hidden_layer_output = tf.nn.sigmoid(hidden_layer_input)
    return hidden_layer_output

# Function to train the neural network
def train_neural_network(X, y, numhidden, vascular_weights, branching_factor=2, bias_factor=0.005, learning_rate=0.001, epochs=1000):

    # Calculate the number of layers based on the tree structure
    num_layers = int(1 + np.ceil(np.log2(numhidden) / np.log2(branching_factor)))

    # Create a list of TreeLayer instances
    tree_layers = [TreeLayer(num_hidden, 2**i if i < num_layers - 1 else numhidden, bias_factor) for i in range(num_layers)]
    
    # Ensure bias_factor is an integer for ClassifierLayer
    bias_factor_classifier = int(round(bias_factor))

    # Create a ClassifierLayer instance
    classifier_layer = ClassifierLayer(numhidden)

    # Initialize neural network weights and biases
    weights = tf.Variable(tf.random.normal((X.shape[1], numhidden)))
    biases = [tf.Variable(tf.zeros((numhidden,)))]
    input_layer = tf.keras.layers.Dense(X.shape[1], activation='linear', use_bias=False)
    # Training loop
    for epoch in range(epochs):
        with tf.GradientTape() as tape:
            # Simulating forward pass of the vascular network
            energy_source = tf.ones(numhidden)
            for layer in tree_layers:
                energy_levels = layer(energy_source)
    
            classifier_output = classifier_layer(input_layer(X[0:1, :]), energy_levels)
    
            # Simulating forward pass of the neural network
            hidden_layer_output = forward_pass_neural_network(energy_levels[-1], weights, biases[0])
    
            # Calculate the loss for both networks
            classifier_loss = tf.reduce_sum(tf.square(classifier_output))  # Replace with actual classifier loss
            neural_network_loss = tf.reduce_sum(tf.square(hidden_layer_output - y[0]))
    
            total_loss = classifier_loss + neural_network_loss

        # Calculate gradients
        gradients = tape.gradient(total_loss, tree_layers + classifier_layer.trainable_variables + [weights, biases[0]])

        # Update vascular weights
        for i in range(num_layers):
            vascular_weights[i].assign_sub(learning_rate * gradients[i])

        # Update classifier weights
        weights.assign_sub(learning_rate * gradients[num_layers])
        biases[0].assign_sub(learning_rate * gradients[num_layers + 1])

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss.numpy()}")

    return weights, biases

if __name__ == "__main__":
    # Load the Iris dataset
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # Convert labels to one-hot encoding
    y_one_hot = tf.keras.utils.to_categorical(y, num_classes=3)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

    num_hidden = 10

    # Initializing vascular weights
    num_nodes = num_hidden * 2  # Assuming branching factor is 2
    vascular_weights = initialize_vascular_weights(2, num_hidden)
    

    # Transpose weights
    weights, biases = train_neural_network(X_train, y_train, num_hidden, vascular_weights)
