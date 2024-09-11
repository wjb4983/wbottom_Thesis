import math
import torch

class ANVN():
    def __init__(self, branching_factor, energy, num_hidden):
        self.branching_factor = branching_factor
        self.energy = energy
        self.root = ANVN_Node(self.branching_factor,False,self.energy,True)
        num_layers = math.ceil(math.log(num_hidden,self.branching_factor))
        def add_layer(parent_node, depth):
            if depth < num_layers - 1:
                parent_node.children = [ANVN_Node(self.branching_factor, False,0, False) for _ in range(self.branching_factor)]
                for child in parent_node.children:
                    add_layer(child, depth + 1)
            else:
                parent_node.children = [ANVN_Node(0, True,0, False) for _ in range(self.branching_factor)]

        add_layer(self.root, 0)
        
import numpy as np
class ANVN_Node():
    def __init__(self, num_children, is_leaf, energy, is_head, verbose=False, max_energy=2):
        self.alpha = 0.1
        self.children = []
        self.energy = energy
        self.is_head = is_head
        self.num_children = num_children
        self.is_leaf = is_leaf
        unnormalized_weights = np.random.rand(num_children)
        self.weights = unnormalized_weights / np.sum(unnormalized_weights)
        self.verbose = verbose
        self.max_energy=max_energy
    def forward(self, energy = None):
        if energy==None:
            energy=self.energy
        # Simulate forward propogation
        # Just calculates what the energy values should be for leafs & intermediate nodes
        if torch.is_tensor(energy):
            if str(energy.device) == 'cuda:0' or str(energy.device) == 'cuda':
                energy = energy.cpu()
            energy = energy.numpy()
        if self.is_leaf:
            # print(energy)
            self.energy=energy
            return np.array([self.energy])
        else:
            self.energy = energy
            # print(energy)
            child_energies = self.weights * energy
            return np.concatenate([child.forward(child_energy) for child, child_energy in zip(self.children, child_energies)])
        
    def backprop(self, train_bias):
        # Currently only uses gradient between hidden node's bias and calculated 
        # what its new bias would be given energy forward prop
        
        #turn into numpy
        if torch.is_tensor(train_bias):
            if str(train_bias.device) == 'cuda:0' or str(train_bias.device) == 'cuda':
                train_bias = train_bias.cpu()
            train_bias = train_bias.numpy()
            
            
        my_energy = self.getenergies()                     #gets array of leaf energies (EL)
        #we are calling gradient 
        # GT - my guess
        #convert "bias" to energy:
        train_bias = 1 + train_bias
        gradient = (train_bias-my_energy) * self.alpha #DELTA = EN-EL
        # energy_gradient = 1 + gradient                     #MATH SAYS TO ADD
        self.setgradient(gradient)    #set gradient
        # for i in range(self.getdepth()):
        self.setenergies()                   #then update energy that the node I am at currently distributes down before normalization
        self.updateweights()  
    def updateweights(self):
        # Just updates the weights based on backprop gradients - not ture anymore
        # Intermediate nodes gradient = average of 
        if self.is_leaf:
            return self.energy
        else:
            #Here we calculate how much energy each child is using and then normalize it between 0 and 1
            child_energies = np.array([child.updateweights() for child in self.children])
            child_energies = child_energies.flatten()
            new_grad = np.average(child_energies)
            # print(child_grad)
            if self.verbose:
                print("before weight",self.weights)
            self.weights = child_energies/np.sum(child_energies)
            if self.verbose:
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
    def setenergies(self, time=0):
        if self.is_leaf:
            # if time == 0:
            #     if verbose:
            #         print("before",self.energy,"after",self.gradient)
            self.energy = np.clip(self.energy+self.gradient, 0, self.max_energy)
            return self.energy
        else:
            self.energy = np.sum([child.setenergies(time) for child in self.children])
            return self.energy
        
    def getdepth(self):
        if self.is_leaf:
            return 0
        else:
            return 1 + self.children[0].getdepth()
    def clip(self):
        self.energy = np.clip(self.energy, 0, self.max_energy)