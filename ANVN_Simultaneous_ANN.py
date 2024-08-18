import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle


verbose=0
num_hidden=512
max_energy = 2

random_seed = 0
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


batch_size = 128

train_dataset = datasets.CIFAR10('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

validation_dataset = datasets.CIFAR10('./data',
                                    train=False,
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)#, generator=torch.Generator(device=device))

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)

import math
class ANVN():
    def __init__(self, branching_factor, energy):
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
    def __init__(self, num_children, is_leaf, energy, is_head):
        self.alpha = 0.1
        self.children = []
        self.energy = energy
        self.is_head = is_head
        self.num_children = num_children
        self.is_leaf = is_leaf
        unnormalized_weights = np.random.rand(num_children)
        self.weights = unnormalized_weights / np.sum(unnormalized_weights)
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
            if verbose:
                print("before weight",self.weights)
            self.weights = child_energies/np.sum(child_energies)
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
    def setenergies(self, time=0):
        if self.is_leaf:
            # if time == 0:
            #     if verbose:
            #         print("before",self.energy,"after",self.gradient)
            self.energy = np.clip(self.energy+self.gradient, 0, max_energy)
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
        self.energy = np.clip(self.energy, 0, max_energy)

class ReLU_Scaler(nn.Module):
    def __init__(self, input_size, energy_init=1.0):
        """
        Initialize the EnergyScaledLayer.
        
        Parameters:
        - input_size (int): Number of input neurons (size of the input tensor).
        - energy_init (float): Initial value for the energy scaling factor.
        """
        super(ReLU_Scaler, self).__init__()
        
        # Initialize energy scaling factors, one for each input neuron
        self.energy = nn.Parameter(torch.full((input_size,), energy_init))

    def forward(self, x):
        """
        Forward pass through the EnergyScaledLayer.
        
        Parameters:
        - x (torch.Tensor): Input tensor with activations from the previous layer.
        
        Returns:
        - torch.Tensor: Output tensor with activations scaled by the energy factors.
        """
        # Element-wise multiplication of input activations with energy scaling factors
        return x * self.energy
    def update_energy(self, new_energy):
        with torch.no_grad():
            self.energy.copy_(new_energy)

class Net(nn.Module):
    def __init__(self, reg_strength=0.01, clip_value=1.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*32*32, num_hidden, bias=False)
        self.fc1_ReLU_scaler = ReLU_Scaler(num_hidden)
        self.fc2 = nn.Linear(num_hidden, 10, bias=False)
        self.clip_value=clip_value


    def forward(self, x):
        x = x.view(-1, 3 * 32 * 32)
        intermediate_output = F.relu(self.fc1(x))  # Intermediate output after first layer
        intermediate_output = self.fc1_ReLU_scaler(intermediate_output)
        x = self.fc2(intermediate_output)
        # return F.log_softmax(x, dim=1), intermediate_output
        return F.log_softmax(x), intermediate_output
    def clip_weights(self):
        # Clip the weights of fc1 and fc2 to be within the range [-clip_value, clip_value]
        for layer in [self.fc1, self.fc2]:
            for param in layer.parameters():
                param.data = torch.clamp(param.data, -self.clip_value, self.clip_value)
    # def normalize_weights(model):
    #     with torch.no_grad():
    #         model.fc1.weight.data /= 10.0
    #         model.fc2.weight.data /= 10.0
    #     self.fc1 = nn.Linear(28 * 28, 1000)
    #     self.fc2 = nn.Linear(1000, 10)
    #     self.reg_strength = reg_strength

    # def forward(self, x):
    #     x = x.view(-1, 28*28)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)

    #     # L2 regularization term
    #     l2_reg = self.reg_strength * (torch.norm(self.fc1.weight) + torch.norm(self.fc2.weight))

    #     return F.log_softmax(x, dim=1) - l2_reg  # Subtract regularization term from output


model = Net(clip_value=99999999.0).to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=.001)
criterion = nn.CrossEntropyLoss()
# print(type(model.children().next()))
ANVN_N = ANVN(2,128)
import numpy as np
ANVN_N.root.forward()
def train(epoch, log_interval=100):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output, _ = model(data)
        
        # Compute loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        optimizer.step()
        
        # Extract gradients for the middle layer (fc1)
        gradients_fc1 = model.fc1.weight.grad.detach().cpu().numpy().mean(1)
        ANVN_N.root.backprop(gradients_fc1)
        ANVN_forward = ANVN_N.root.forward()
        ANVN_forward = torch.tensor(ANVN_forward)
        model.fc1_ReLU_scaler.update_energy(ANVN_forward)
        model.clip_weights()
        if batch_idx % log_interval == 0:
            # print(ANVN_N.root.forward())
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.data.item()))
        
def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output,_ = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        model.clip_weights()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))



epochs = 25

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)
print(ANVN_N.root.forward())
with open('ANVN.pkl', 'wb') as f:
    pickle.dump(ANVN_N, f)

torch.save(model.state_dict(), 'trained_model_cf_256_test!!!.pt')