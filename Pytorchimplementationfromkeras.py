import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import random
import os
import PySNN

verbose = False

class CustomHidden(nn.Module):
    def __init__(self, input_size, num_hidden, activation='sigmoid'):
        super(CustomHidden, self).__init__()
        self.units = num_hidden
        self.kernel_regularizer = nn.Linear(input_size, num_hidden, bias=False)
        self.bias = nn.Parameter(torch.rand(num_hidden))
        self.activation = getattr(torch.nn.functional, activation)

    def forward(self, inputs):
        return self.activation(torch.mm(inputs, self.kernel_regularizer.weight.T) - self.bias)

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
        self.alpha = .1#/50
        self.is_head = is_head

    def forward(self, energy):
        # Simulate forward propogation
        # Just calculates what the energy values should be for leafs & intermediate nodes
        if self.is_leaf:
            # print(energy)
            self.energy =np.clip(energy, 0, 2)
            return np.array([self.energy])
        else:
            self.energy = energy
            # print(energy)
            child_energies = self.weights * energy
            return np.concatenate([child.forward(child_energy) for child, child_energy in zip(self.children, child_energies)])
        
        
    def backprop(self, train_bias):
        # Currently only uses gradient between hidden node's bias and calculated 
        # what its new bias would be given energy forward prop
        my_energy = self.getenergies()                     #gets array of leaf energies (EL)
        gradient = ((1-train_bias)-my_energy) * self.alpha #DELTA = EN-EL
        # energy_gradient = 1 + gradient                     #MATH SAYS TO ADD
        if verbose:
            # print("my energy",my_energy)
            print("grad",gradient)
            # print("my",my_bias)
            # print("train",train_bias)
        self.setgradient(gradient)    #set gradient
        for i in range(self.getdepth()):
            self.setenergies(i)                   #then update energy that the node I am at currently distributes down before normalization
        self.updateweights()             

    def updateweights(self):
        # Just updates the weights based on backprop gradients - not ture anymore
        # Intermediate nodes gradient = average of 
        if self.is_leaf:
            return self.energy
        else:
            child_grad = np.array([child.updateweights() for child in self.children])
            child_grad = child_grad.flatten()
            # print(self.weights.size)
            # print(child_grad.size)
            new_grad = np.average(child_grad)
            # print(child_grad)
            if verbose:
                print("before weight",self.weights)
            self.weights = child_grad/np.sum(child_grad)
            if verbose:
                print("after weight",self.weights)

            # This might do something?
            # This is supposed to make the nodes coming from the head node have
            # Custom weights proportional to the energy taken
            # so the weights can add up to < head.energy
            # However once backprop is done, this never happens
            if self.is_head:
                total_used_energy = np.sum(self.getenergies())
                self.weights = self.weights*(total_used_energy/self.energy)
            return new_grad
            
    #sets gradient at leaf nodes
    def setgradient(self, grad):
        # Takes gradient of each leaf calculated and puts it in the right leaf
        # Doesn't give gradient for the intermediate nodes
        if self.is_leaf:
            self.gradient = grad
        else:
            # print(grad)
            # print(self.num_children)
            childgrad = np.split(grad, self.num_children)
            for child, child_gradient in zip(self.children, childgrad):
                child.setgradient(child_gradient)
            
    #gets sum of all energies at leaf
    def checksum(self):
        if self.is_leaf:
            return np.abs(self.energy)
        else:
            return np.sum([child.checksum() for child in self.children])
        
    #gets array of all energies at leaf
    def getenergies(self):
        if self.is_leaf:
            return np.array([self.energy])
        else:
            return np.concatenate([child.getenergies() for child in self.children])
        
    #Updates energies at leaf using the gradient
    def setenergies(self, time):
        if self.is_leaf:
            if time == 0:
                if verbose:
                    print("before",self.energy,"after",self.gradient)
                self.energy = np.clip(self.energy+self.gradient, 0, 2)
            return self.energy
        else:
            self.energy = np.sum([child.setenergies(time) for child in self.children])
            return self.energy
        
    def getdepth(self):
        if self.is_leaf:
            return 0
        else:
            return 1 + self.children[0].getdepth()
            

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


def reset_random_seeds(n):
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)


reset_random_seeds(40)

# For our runs with everything going:
# This will be set up as [numhidden, nunm_children_per_node]
# We will run this loop on all 3 datasets
loop_iter = [[4,2], [4,4], [16,2], [16,4], [32, 2], [64, 2], [128, 2], [256,2], [512,2]]

#set up vascular tree w/ params
# The number of hidden nodes that the classifier will have
# Note it will only have one layer for now
numhidden = 16

num_children_per_node = 2

vascular_tree = VascularTree(numhidden, num_children_per_node)

####################################################################
##need to train classifier to extract biases from:::
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


# Create MLP without any vascular stuff and run tests
input_size = X_train_IRIS.shape[1]
output_size = len(set(y_train_IRIS))

class MLP(nn.Module):
    def __init__(self, input_size, num_hidden):
        super(MLP, self).__init__()
        self.layers = nn.Sequential(
            # nn.Flatten(),
            # nn.Linear(input_size, input_size),
            CustomHidden(input_size, num_hidden),
            nn.Linear(num_hidden,output_size),
            nn.Softmax(dim=1))

    def forward(self, x):
        return self.layers(x)

model = MLP(input_size, numhidden)

# Criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
epochs = 50
for epoch in range(epochs):
    inputs = torch.tensor(X_train_IRIS, dtype=torch.float32)
    labels = torch.tensor(y_train_IRIS, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

# Evaluate on the test set
with torch.no_grad():
    inputs_test = torch.tensor(X_test_IRIS, dtype=torch.float32)
    labels_test = torch.tensor(y_test_IRIS, dtype=torch.long)
    outputs_test = model(inputs_test)
    _, predicted = torch.max(outputs_test, 1)
    accuracy = (predicted == labels_test).sum().item() / len(labels_test)
    print(f"Accuracy on the test set: {accuracy}")

# Extract biases
hidden_biases = model.layers[0].bias.detach().numpy()
sum_biases = np.sum(1 - hidden_biases)
if verbose:
    print("Original biases:", hidden_biases)
    print("Sum of original biases:", sum_biases)

# Training the vascular tree
for epoch in range(66):
    if verbose:
        print(f"=============EPOCH {epoch}====================")
    tree_output = vascular_tree.root.forward(sum_biases)
    tree_output = np.maximum(0, tree_output)
    tree_output = np.minimum(2, tree_output)
    vascular_tree.root.backprop(hidden_biases)
    if verbose:
        print("Average gradient:", np.sum(np.abs(hidden_biases - (1 - tree_output))) / numhidden)

if verbose:
    print("Checksum:", vascular_tree.root.checksum())

# Normalize energy -> bias
tree_output = 1 - tree_output
tree_output[tree_output < -1] = -1
if verbose:
    print("Tree output values:", tree_output)
    print("Gradient:", hidden_biases - tree_output)

# Rerun classifier with new biases
model.layers[0].bias.data = torch.tensor(tree_output, dtype=torch.float32)

# Evaluate on the test set with updated biases
with torch.no_grad():
    inputs_test = torch.tensor(X_test_IRIS, dtype=torch.float32)
    labels_test = torch.tensor(y_test_IRIS, dtype=torch.long)
    outputs_test = model(inputs_test)
    _, predicted = torch.max(outputs_test, 1)
    accuracy = (predicted == labels_test).sum().item() / len(labels_test)
    print(f"Accuracy on the test set with energy: {accuracy}")
