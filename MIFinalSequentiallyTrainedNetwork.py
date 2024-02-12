#Works - full script in MIFinal-runallscript.py
# This has a functional vasc and classifier
# vasc still is eh - the back prop needs work somehow



import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
from tensorflow.keras.constraints import Constraint
import tensorflow.keras.backend as K

# This prevents the biases from becoming too large in pos and neg direction
# doesn't seem necessary as biases are already in that range for some reason
# class BiasConstraint(Constraint):
#     def __init__(self, min_value=None, max_value=None):
#         self.min_value = min_value
#         self.max_value = max_value

#     def __call__(self, w):
#         if self.min_value is not None:
#             w = K.maximum(w, self.min_value)
#         if self.max_value is not None:
#             w = K.minimum(w, self.max_value)
#         return w

#     def get_config(self):
#         return {'min_value': self.min_value, 'max_value': self.max_value}

# keep for later just in case
# class NonNegBiasConstraint(tf.keras.constraints.Constraint):
#     def __call__(self, w):
#         return tf.nn.relu(w)


# The custom hidden layer for the classifier
# 
class CustomHidden(tf.keras.layers.Layer):
    def __init__(self, num_hidden, activation='sigmoid', **kwargs):
        super(CustomHidden, self).__init__(**kwargs)
        # Make the barely custom/modified hidden classifier layer
        # Really only important thing is making sure bias is subtracted in the equation
        self.units = num_hidden
        self.kernel_regularizer = tf.keras.regularizers.get(regularizers.l2(0.01))
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        # I regularized weights to make sure they didnt become super powerful and overpower
        # the bias
        self.kernel = self.add_weight(
            "kernel", (input_shape[1], self.units),
            regularizer=self.kernel_regularizer,
        )
        self.bias = self.add_weight("bias", (self.units,))#, constraint=BiasConstraint(min_value=-1.0, max_value=1.0))
        super(CustomHidden, self).build(input_shape)

    def call(self, inputs):
        # We SUBTRACT bias since bias = 1-energy so ENERGY increase = bias decrease
        # decrease in negative bias is an increase
        # increase in sigmoid is increase in probability of firing
        return self.activation(tf.matmul(inputs, self.kernel) - self.bias)




# Singular node of the tree
# Provides most of the methods for SEQUENTIAL TRAINING
# In order to do simultaneous and/or with reservoir, need special special classifier
# So that we can do backprop the right way
class VascularTreeNode:
    def __init__(self, num_children, is_leaf, energy, is_head):
        # Weights are random at first - this is ok
        # I keep too many variables here - change some to global or somethign later
        self.num_children = num_children
        self.is_leaf = is_leaf
        unnormalized_weights = np.random.rand(num_children)
        self.weights = unnormalized_weights / np.sum(unnormalized_weights)
        self.children = []
        self.energy = energy;
        self.gradient = 0;
        self.alpha = 2
        self.is_head = is_head

    def forward(self, energy):
        # Simulate forward propogation
        # Just calculates what the energy values should be for leafs & intermediate nodes
        if self.is_leaf:
            self.energy = energy
            # print(energy)
            return np.array([self.energy])
        else:
            self.energy = energy
            # print(energy)
            child_energies = self.weights * energy
            return np.concatenate([child.forward(child_energy) for child, child_energy in zip(self.children, child_energies)])
        
        
    def backprop(self, train_bias):
        # Currently only uses gradient between hidden node's bias and calculated 
        # what its new bias would be given energy forward prop
        my_energy = self.getenergies()
        gradient = (train_bias-(1-my_energy)) * self.alpha
        energy_gradient = 1- gradient
        # print("grad",gradient)
        # print("my",my_bias)
        # print("train",train_bias)
        self.setgradient(energy_gradient)
        self.updateweights()

    def updateweights(self):
        # Just updates the weights based on backprop gradients
        # Intermediate nodes gradient = average of 
        if self.is_leaf:
            return self.gradient
        else:
            child_grad = np.array([child.updateweights() for child in self.children])
            child_grad = child_grad.flatten()
            # print(self.weights.size)
            # print(child_grad.size)
            new_grad = np.average(child_grad)
            self.weights -= child_grad
            self.weights = self.weights/np.sum(self.weights)
            # This might do something?
            # This is supposed to make the nodes coming from the head node have
            # Custom weights proportional to the energy taken
            # so the weights can add up to < head.energy
            # However once backprop is done, this never happens
            if self.is_head:
                total_used_energy = np.sum(self.getenergies())
                self.weights = self.weights*(total_used_energy/self.energy)
            return new_grad
            
    def setgradient(self, grad):
        # Takes gradient of each leaf calculated and puts it in the right leaf
        # Doesn't give gradient for the intermediate nodes
        if self.is_leaf:
            self.gradient = grad
            return
        else:
            # print(grad)
            # print(self.num_children)
            childgrad = np.split(grad, self.num_children)
            for child, child_gradient in zip(self.children, childgrad):
                child.setgradient(child_gradient)
            
        
    def checksum(self):
        if self.is_leaf:
            return np.abs(self.energy)
        else:
            return np.sum([child.checksum() for child in self.children])
    def getenergies(self):
        if self.is_leaf:
            return np.array([self.energy])
        else:
            return np.concatenate([child.getenergies() for child in self.children])
            

class VascularTree:
    def __init__(self, numhidden, num_children_per_node):
        self.root = VascularTreeNode(num_children_per_node, False,100, True)
        num_layers = math.ceil(math.log(numhidden,num_children_per_node))
        def add_layer(parent_node, depth):
            if depth < num_layers - 1:
                parent_node.children = [VascularTreeNode(num_children_per_node, False,0, False) for _ in range(num_children_per_node)]
                for child in parent_node.children:
                    add_layer(child, depth + 1)
            else:
                parent_node.children = [VascularTreeNode(0, True,0, False) for _ in range(num_children_per_node)]

        add_layer(self.root, 0)

class VascularClassifier:
    def __init__(self, hidden_size, hidden_biases):
        # self.weights_hidden = np.random.randn(input_size, hidden_size)
        self.bias_hidden = hidden_biases
        # self.weights_output = np.random.randn(hidden_size, output_size)
        self.bias_output = np.zeros((1, hidden_size))

    def forward(self, input_values):
        # output_layer = np.dot(hidden_layer, self.weights_output) + self.bias_output
        return input_values #output_layer
    
def trainWithDataSet(X_train, Y_train, X_test, Y_test):
    pass

# For our runs with everything going:
# This will be set up as [numhidden, nunm_children_per_node]
# We will run this loop on all 3 datasets
loop_iter = [[4,2], [4,4], [16,2], [16,4], [32, 2], [64, 2], [128, 2], [256,2], [512,2]]

#set up vascular tree w/ params
# The number of hidden nodes that the classifier will have
# Note it will only have one layer for now
numhidden = 16

num_children_per_node = 2

tf.random.set_seed(1)

vascular_tree = VascularTree(numhidden, num_children_per_node)

####################################################################
##need to train classifier to extract biases:::
    ################################################################
    
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train_IRIS, X_test_IRIS, y_train_IRIS, y_test_IRIS = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_IRIS = scaler.fit_transform(X_train_IRIS)
X_test_IRIS = scaler.transform(X_test_IRIS)


##########
#first, Create MLP without any vascular stuff and run da tests
#IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS IRIS 
##########

# These get used to tell the model how to shape for the IRIS DATASET <----

input_size = X_train_IRIS.shape[1]
output_size = len(set(y_train_IRIS))

# Input layer
input_layer = tf.keras.layers.Input(shape=(input_size,))

# Hidden layer with trainable weights and biases
hidden_layer = CustomHidden(numhidden, trainable=True)(input_layer)
# Example call if we want bias to be constrained
# hidden_layer = CustomHidden(numhidden, trainable=True, bias_constraint=BiasConstraint(min_value=0.0, max_value=2.0))(input_layer)

# Softmax output layer
output_layer = tf.keras.layers.Dense(output_size, activation='softmax')(hidden_layer)

# Create the model
model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
# Need to silince when we do many runs
history = model.fit(X_train_IRIS, y_train_IRIS, epochs=50, batch_size=1, verbose=0)#, validation_data=(X_test, y_test))

#############################################################################################
# now we perform the test on the MLP w/ NO VASCULAR NETWORK

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Split the dataset into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=100)

accuracy = model.evaluate(X_test_IRIS, y_test_IRIS, verbose=0)[1]
print(f"Accuracy on the test set: {accuracy}")


# Weights of entire hidden layer
all_weights = model.layers[1].get_weights()

##############################################################################################
##############################################################################################

# Now we extract biases, and get to 
hidden_biases = model.layers[1].get_weights()[1]
classifier = VascularClassifier(numhidden, hidden_biases)

print("orig biases: ", hidden_biases)
sum_biases = np.sum(np.abs(1+hidden_biases))
print("sum of orig biases", sum_biases)
# print("sanity check biases: ", classifier.bias_hidden)
# tree_output = vascular_tree.root.forward(vascular_tree.root.energy)

for epoch in range(50):
    tree_output = vascular_tree.root.forward(sum_biases)
    tree_output = np.maximum(0, tree_output)
    tree_output = np.minimum(2,tree_output)
    tree_backprop = vascular_tree.root.backprop(hidden_biases)

print("checksum: ", vascular_tree.root.checksum())


# print("Input values:", vascular_tree.root.energy)
# print("Tree output values:", tree_output)
# Normalize energy -> bias
tree_output = 1 - tree_output
tree_output[tree_output < -1] = -1

print("grad:",hidden_biases-tree_output)



# Now rerun classifier with new biases
all_weights[1] = tree_output
model.layers[1].set_weights(all_weights)
del tree_output

accuracy = model.evaluate(X_test_IRIS, y_test_IRIS, verbose=0)[1]
print(f"Accuracy on the test set with energy: {accuracy}")
# model.summary()
