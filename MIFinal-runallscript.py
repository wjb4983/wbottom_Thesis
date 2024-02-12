import numpy as np
import tensorflow as tf
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

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
        self.alpha = .01
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
        # print("grad",gradient)
        # print("my",my_bias)
        # print("train",train_bias)
        self.setgradient(gradient)
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
            
#########################################################################################
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

    ##############################################################33

# For our runs with everything going:
# This will be set up as [numhidden, nunm_children_per_node]
# We will run this loop on all 3 datasets
loop_iter = [[4,2],[64,2],[128,2],[256,2],[512,2]]#[[4,2], [16,2]], [32, 2], [64, 2], [128, 2], [256,2], [512,2]]
# loop_iter = [[4,2]] #,[4,4],[128, 2], [512,2]]

# IRIS DATASET:
# Split the dataset into training and testing sets
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target  
X_train_IRIS, X_test_IRIS, y_train_IRIS, y_test_IRIS = train_test_split(X, y, test_size=0.2, random_state=42)
# Standardize the features
scaler = StandardScaler()
X_train_IRIS = scaler.fit_transform(X_train_IRIS)
X_test_IRIS = scaler.transform(X_test_IRIS)


from tensorflow.keras.datasets import mnist
# MNIST DATASET:
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

num_samples = 120
train_images_subset = train_images[:num_samples]
train_labels_subset = train_labels[:num_samples]


# Normalize pixel values to be between 0 and 1
train_images_subset = train_images_subset / 255.0
test_images = test_images / 255.0




# Flatten the input images using NumPy
train_images_flat = train_images_subset.reshape((num_samples, -1))
test_images_flat = test_images.reshape((test_images.shape[0], -1))



# [train x, train y, test x, test y, num input features, num outputs (num classes)]
datasets = [[X_train_IRIS, y_train_IRIS,X_test_IRIS,  y_test_IRIS, 4, 3, "IRIS"], [train_images_flat, train_labels_subset, test_images_flat, test_labels, 28*28, 10, "MNIST"]]

# [amount of root energy per run]
energy_loop = [1,2,4, 100]

# simple for quick run
# energy_loop = [100]

# [no vasc acc, random vasc acc, sequential vasc acc]
output_data = []
print("Here we go")


tf.random.set_seed(1)
for data in datasets:
    for loopdata in loop_iter:
        X_train = data[0]
        Y_train = data[1]
        X_test = data[2]
        Y_test = data[3]
        input_size = data[4]
        output_size = data[5]
        numhidden = loopdata[0]
        num_children_per_node = loopdata[1]
        ##########
        #first, Create MLP without any vascular stuff and run da tests
        ##########
        
        # input layer
        input_layer = tf.keras.layers.Input(shape=(input_size,))
        
        # custon hidden layer with trainable weights and biases
        hidden_layer = CustomHidden(numhidden, trainable=True)(input_layer)
        # Example call if we want bias to be constrained
        # hidden_layer = CustomHidden(numhidden, trainable=True, bias_constraint=BiasConstraint(min_value=0.0, max_value=2.0))(input_layer)
        
        # softmax output layer
        output_layer = tf.keras.layers.Dense(output_size, activation='softmax')(hidden_layer)
        
        # create da model
        model = tf.keras.models.Model(inputs=input_layer, outputs=output_layer)
        print("Compiling Model", end='\r')
        # Compile the model
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("\033[K", end='')  # This clears the line
        print("Training Model", end='\r')
        # Train the model
        # Need to silince when we do many runs
        history = model.fit(X_train, Y_train, epochs=400, batch_size=16, verbose=0)#, validation_data=(X_test, y_test))
        weights = model.layers[1].get_weights()
        # get this now
        accuracy_with_novasc = model.evaluate(X_test, Y_test, verbose=0)[1]
        orig_energy = np.sum(1- weights[1])
        for energy in energy_loop:
            model.layers[1].set_weights(weights)
            print("\033[K", end='')
            print(f"\rLoop {data[6]} energy={energy} loopdata: {loopdata}", flush=True)
    
            #set up vascular tree w/ params
            # The number of hidden nodes that the classifier will have
            # Note it will only have one layer for now
            vascular_tree = VascularTree(numhidden, num_children_per_node)
            
            print("\033[K", end='')  # This clears the line
            # Weights of entire hidden layer
            all_weights = model.layers[1].get_weights()
            
            ##############################################################################################
            ##############################################################################################
            
            #############################################################################################
            # now we perform the test on the MLP w/ RANDOM VASCULAR
            # If we dont train the network, why make it?
            # Just create random values between -1 and 1
            print("\033[K", end='')  # This clears the line
            print("Evaulating Model w/ random Bias", end='\r')
            all_weights[1] = tf.random.uniform(shape=(numhidden,), minval=-1, maxval=1)
            model.layers[1].set_weights(all_weights)
            
            accuracy_with_ranvasc = model.evaluate(X_test, Y_test, verbose=0)[1]
            
            ##############################################################################################
            ##############################################################################################
            # Now we extract biases, and get to 
            hidden_biases = model.layers[1].get_weights()[1]
            # print("orig biases: ", hidden_biases)
            sum_biases = np.sum(np.abs(hidden_biases))
            # print("sum of orig biases", sum_biases)
            # print("sanity check biases: ", classifier.bias_hidden)
            # tree_output = vascular_tree.root.forward(vascular_tree.root.energy)
            print("\033[K", end='')  # This clears the line
            print("Simulating Backprop w/ Neurovascular", end='\r')
            for epoch in range(100):
                tree_output = vascular_tree.root.forward(energy)
                tree_output = np.maximum(0, tree_output)
                tree_output = np.minimum(2,tree_output)
                tree_backprop = vascular_tree.root.backprop(hidden_biases)
            
            # print("checksum: ", vascular_tree.root.checksum())
            
            
            # print("Input values:", vascular_tree.root.energy)
            # print("Tree output values:", tree_output)
            # Normalize energy -> bias
            tree_output = 1 - tree_output
            tree_output[tree_output < -1] = -1
            # Now rerun classifier with new biases
            all_weights[1] = tree_output
            model.layers[1].set_weights(all_weights)
            del tree_output
            print("\033[K", end='')  # This clears the line
            print("Evaluating Model w/ Neurovascular ", end='\r')
            accuracy_with_tvasc = model.evaluate(X_test, Y_test, verbose=0)[1]
            # print(f"Accuracy on the test set with energy: {accuracy_with_tvasc}")
            # model.summary()
            # [3 accs, numhidden, numchild, name of dataset, energy, orig energy]
            acc_array = [accuracy_with_novasc, accuracy_with_ranvasc , accuracy_with_tvasc, numhidden, num_children_per_node, data[6], energy, orig_energy]
            output_data.append(acc_array)
            #delete everything from memory
            #python probably does this
            #but we are gonna do it for safety of my memory
            del all_weights
            print("\033[K", end='')  # This clears the line

print("\033[K", end='')  # This clears the line
print('\r')
print("done")

# split data
output_data = np.array(output_data)
IRIS_DATA = output_data[output_data[:, 5] == "IRIS"]
MNIST_DATA = output_data[output_data[:, 5] == "MNIST"]


IRIS_DATA = sorted(IRIS_DATA, key=lambda x: x[3])
MNIST_DATA = sorted(MNIST_DATA, key=lambda x: x[3])

IRIS_DATA = np.array(IRIS_DATA)
MNIST_DATA = np.array(MNIST_DATA)


plot_data_IRIS = IRIS_DATA[:, [0, 1, 2]].astype(float)
plot_data_MNIST = MNIST_DATA[:, [0, 1, 2]].astype(float)

label_list = ['Basic Classifier', 'Classifier with Randomized Vascular Network', 'Classifier with Sequentially Trained Vascular Network']
markers = ['o', 'x', '^', 'v', '<', '>', 'D', 'P']
colors = ['green', 'red', 'black']


#fun index things to extract data to be in seperate energy plots but vary num hidden on same plot
energy_values = IRIS_DATA[:, 6]
numeric_energy_mask = np.char.isnumeric(energy_values)
numeric_energy_values = energy_values[numeric_energy_mask].astype(int)
unique_energy_values = np.unique(numeric_energy_values)
for energy_value in unique_energy_values:
    # Create a new plot for each energy value
    plt.figure()


    # print(f"Energy Mask: {energy_mask}")

    for i in range(3):
        # Sort the indices based on the x-values
        sorted_indices = np.argsort(IRIS_DATA[numeric_energy_mask, 3].astype(int))
        sorted_x_values = IRIS_DATA[numeric_energy_mask, 3].astype(int)[sorted_indices]
        sorted_y_values = plot_data_IRIS[numeric_energy_mask, i][sorted_indices]

        # Filter data for the current energy value
        energy_mask = IRIS_DATA[numeric_energy_mask, 6].astype(int) == energy_value
        


        # Scatter plot with different shapes for each num_hidden
        plt.scatter(IRIS_DATA[numeric_energy_mask][energy_mask, 3].astype(int),
                    plot_data_IRIS[numeric_energy_mask][energy_mask, i],
                    marker=markers[i], color=colors[i], label=label_list[i])

    plt.title(f'IRIS Dataset - Number of Hidden Neurons vs Accuracy (Energy={energy_value})')
    plt.xlabel('numhidden')
    plt.ylabel('Accuracy')
    plt.gca().set_ylim([0, 1])
    plt.legend()
    plt.savefig(f"MIFinal-IRIS-{numhidden}-{energy_value}.png")

# Show the plots
plt.show()

#fun index things to extract data to be in seperate energy plots but vary num hidden on same plot
energy_values = MNIST_DATA[:, 6]
numeric_energy_mask = np.char.isnumeric(energy_values)
numeric_energy_values = energy_values[numeric_energy_mask].astype(int)
unique_energy_values = np.unique(numeric_energy_values)
for energy_value in unique_energy_values:
    # Create a new plot for each energy value
    plt.figure()
    # Iterate over each accuracy
    for i in range(3):
        # Sort the indices based on the x-values
        sorted_indices = np.argsort(MNIST_DATA[numeric_energy_mask, 3].astype(int))
        sorted_x_values = MNIST_DATA[numeric_energy_mask, 3].astype(int)[sorted_indices]
        sorted_y_values = plot_data_MNIST[numeric_energy_mask, i][sorted_indices]

        # Filter data for the current energy value
        energy_mask = MNIST_DATA[numeric_energy_mask, 6].astype(int) == energy_value
        


        # Scatter plot with different shapes for each num_hidden
        plt.scatter(MNIST_DATA[numeric_energy_mask][energy_mask, 3].astype(int),
                    plot_data_MNIST[numeric_energy_mask][energy_mask, i],
                    marker=markers[i], color=colors[i], label=label_list[i])

    plt.title(f'MNIST Dataset - Number of Hidden Neurons vs Accuracy (Energy={energy_value})')
    plt.xlabel('numhidden')
    plt.ylabel('Accuracy')
    plt.gca().set_ylim([0, 1])
    plt.legend()
    plt.savefig(f"MIFinal-MNIST-{numhidden}-{energy_value}.png")


# Show the plots
plt.show()

# # Code to print energy data:
#############################
# energy_values = IRIS_DATA[:, 7].astype(float)


# numeric_energy_mask = np.isfinite(energy_values)
# numeric_energy_values = energy_values[numeric_energy_mask]
# unique_energy_values = np.unique(numeric_energy_values)

# for original_energy_value in unique_energy_values:
#     # Find indices where the original energy value matches
#     indices = np.where(numeric_energy_values == original_energy_value)[0]
    
#     # Get the corresponding new energy values
#     corresponding_new_energy_values = IRIS_DATA[numeric_energy_mask][indices, 6]
    
#     # Print the results
#     print(f"Original Energy Value: {original_energy_value}")
#     print(f"Corresponding New Energy Values: {corresponding_new_energy_values}")
#     print(f"Corresponding num hiddens: {IRIS_DATA[numeric_energy_mask][indices, 3]}")
#     print()
