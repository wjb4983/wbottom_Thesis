import torch
from bindsnet.conversion import ann_to_snn
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from bindsnet.network.monitors import Monitor
from time import time as t_
import pandas as pd
import os
from matplotlib.ticker import MultipleLocator
import numpy as np

batch_size=256
num_hidden=256
verbose=0


from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from typing import Iterable, Optional, Union
from bindsnet.network import nodes
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
            # self.energy =np.clip(energy, 0, 1)
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
        gradient = (train_bias-my_energy) * self.alpha #DELTA = EN-EL
        # energy_gradient = 1 + gradient                     #MATH SAYS TO ADD
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
    def setenergies(self, time):
        if self.is_leaf:
            # if time == 0:
            #     if verbose:
            #         print("before",self.energy,"after",self.gradient)
            #     self.energy = np.clip(self.energy+self.gradient, 0, 2)
            return self.energy
        else:
            self.energy = np.sum([child.setenergies(time) for child in self.children])
            return self.energy
        
    def getdepth(self):
        if self.is_leaf:
            return 0
        else:
            return 1 + self.children[0].getdepth()
        
# class SubtractiveResetIFNodes(nodes.Nodes):
#     # language=rst
#     """
#     Layer of `integrate-and-fire (IF) neurons <https://bit.ly/2EOk6YN>` using
#     reset by subtraction.
#     """

#     def __init__(
#         self,
#         n: Optional[int] = None,
#         shape: Optional[Iterable[int]] = None,
#         traces: bool = False,
#         traces_additive: bool = False,
#         tc_trace: Union[float, torch.Tensor] = 20.0,
#         trace_scale: Union[float, torch.Tensor] = 1.0,
#         sum_input: bool = False,
#         thresh: Union[float, torch.Tensor] = -52.0,
#         reset: Union[float, torch.Tensor] = -65.0,
#         refrac: Union[int, torch.Tensor] = 5,
#         lbound: float = None,
#         **kwargs,
#     ) -> None:
#         # language=rst
#         """
#         Instantiates a layer of IF neurons with the subtractive reset mechanism
#         from `this paper <https://bit.ly/2ShuwrQ>`_.

#         :param n: The number of neurons in the layer.
#         :param shape: The dimensionality of the layer.
#         :param traces: Whether to record spike traces.
#         :param traces_additive: Whether to record spike traces additively.
#         :param tc_trace: Time constant of spike trace decay.
#         :param trace_scale: Scaling factor for spike trace.
#         :param sum_input: Whether to sum all inputs.
#         :param thresh: Spike threshold voltage.
#         :param reset: Post-spike reset voltage.
#         :param refrac: Refractory (non-firing) period of the neuron.
#         :param lbound: Lower bound of the voltage.
#         """
#         super().__init__(
#             n=n,
#             shape=shape,
#             traces=traces,
#             traces_additive=traces_additive,
#             tc_trace=tc_trace,
#             trace_scale=trace_scale,
#             sum_input=sum_input,
#         )

#         self.register_buffer(
#             "reset", torch.tensor(reset, dtype=torch.float)
#         )  # Post-spike reset voltage.
#         self.register_buffer(
#             "thresh", torch.tensor(thresh, dtype=torch.float)
#         )  # Spike threshold voltage.
#         self.register_buffer(
#             "refrac", torch.tensor(refrac)
#         )  # Post-spike refractory period.
#         self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
#         self.register_buffer(
#             "refrac_count", torch.FloatTensor()
#         )  # Refractory period counters.

#         self.lbound = lbound  # Lower bound of voltage.

#     def forward(self, x: torch.Tensor) -> None:
#         # language=rst
#         """
#         Runs a single simulation step.

#         :param x: Inputs to the layer.
#         """
#         # Integrate input voltages.
#         self.v += (self.refrac_count == 0).float() * x

#         # Decrement refractory counters.
#         self.refrac_count = (self.refrac_count > 0).float() * (
#             self.refrac_count - self.dt
#         )

#         # Check for spiking neurons.
#         self. s = self.v >= self.thresh

#         # Refractoriness and voltage reset.
#         self.refrac_count.masked_fill_(self.s, self.refrac)
#         self.v[self.s] = self.v[self.s] - self.thresh

#         # Voltage clipping to lower bound.
#         if self.lbound is not None:
#             self.v.masked_fill_(self.v < self.lbound, self.lbound)

#         super().forward(x)

#     def reset_state_variables(self) -> None:
#         # language=rst
#         """
#         Resets relevant state variables.
#         """
#         super().reset_state_variables()
#         self.v.fill_(self.reset)  # Neuron voltages.
#         self.refrac_count.zero_()  # Refractory period counters.

#     def set_batch_size(self, batch_size) -> None:
#         # language=rst
#         """
#         Sets mini-batch size. Called when layer is added to a network.

#         :param batch_size: Mini-batch size.
#         """
#         super().set_batch_size(batch_size=batch_size)

#         device = self.reset.device
#         # shape_tensor = torch.tensor(self.shape, device=device)
#         self.v.to(device)
#         # print((self.v.device), (self.reset.device), (self.shape))
#         self.v = self.reset * torch.ones(batch_size, *self.shape, device=self.reset.device)
        
#         self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
class SubtractiveResetIFNodes(nodes.Nodes):
    """
    Layer of integrate-and-fire (IF) neurons using reset by subtraction.
    """

    def __init__(
        self,
        n: Optional[int] = None,
        shape: Optional[Iterable[int]] = None,
        traces: bool = False,
        traces_additive: bool = False,
        tc_trace: Union[float, torch.Tensor] = 20.0,
        trace_scale: Union[float, torch.Tensor] = 1.0,
        sum_input: bool = False,
        thresh: Union[float, torch.Tensor] = -52.0,
        reset: Union[float, torch.Tensor] = -65.0,
        refrac: Union[int, torch.Tensor] = 5,
        lbound: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        # self.traces = traces
        # self.traces_additive = traces_additive
        # self.tc_trace = tc_trace
        # self.trace_scale = trace_scale
        # self.sum_input = sum_input

        self.register_buffer("reset", torch.tensor(reset, dtype=torch.float))
        self.register_buffer("thresh", torch.tensor(thresh, dtype=torch.float))
        self.register_buffer("refrac", torch.tensor(refrac, dtype=torch.float))
        self.register_buffer("v", torch.FloatTensor())
        self.register_buffer("refrac_count", torch.FloatTensor())

        self.lbound = lbound


    def forward(self, x: torch.Tensor) -> None:
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
        # print((self.thresh ==1).sum())
        if self.v.dim() != x.dim():
            raise ValueError("Input dimensions must match the neuron state dimensions")

        # Integrate input voltages
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters
        self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)

        # Check for spiking neurons
        self.s = self.v >= self.thresh
        thresh_tensor = self.thresh.unsqueeze(0).expand_as(self.v)

        # Refractoriness and voltage reset
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v[self.s] -= thresh_tensor[self.s]

        # Voltage clipping to lower bound
        if self.lbound is not None:
            self.v = torch.max(self.v, self.lbound)

        super().forward(x)

    def reset_state_variables(self) -> None:
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.reset)
        self.refrac_count.zero_()

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)

        device = self.reset.device
        # Ensure self.thresh is initialized with the correct size
        if self.thresh.dim() == 0:# and self.thresh.shape != (self.n):
            
            self.thresh = self.thresh.expand(self.n)
        elif self.thresh.dim() == 1 and self.thresh.size(0) != self.n:
            raise ValueError(f"Expected threshold tensor of size {self.n}, but got {self.thresh.size(0)}")

        # Initialize voltages and refractory counters
        self.v = self.reset * torch.ones(batch_size, *self.shape, device=device)
        self.refrac_count = torch.zeros_like(self.v, device=device)

class ANVN_SRIFNodes(SubtractiveResetIFNodes):
    def __init__(self, *args, spike_limit=1000, device = 'cuda', batch_size=batch_size, **kwargs):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device
        self.spike_counts = torch.zeros((self.batch_size,1), device=self.device)
        self.tot_spikes=0
        self.spike_limit = spike_limit
        self.original_thresh = self.thresh
        self.energy_usage = torch.zeros((self.batch_size, self.n))

        
        

    def forward(self, *args, **kwargs):
        # Increment spike count for each batch element
        #Keep this in case I want to do some wierd spike limiting
        if self.energy_usage is None or self.energy_usage.size(0) != self.s.shape[0]:
            self.energy_usage = torch.zeros((self.s.shape[0], self.n), device=self.device)
        self.energy_usage += self.s
        # self.spike_counts += self.s.sum(dim=1, keepdim=True)
        # self.tot_spikes+=self.spike_counts.sum()
        # Check if spike limit reached for any batch element
        # exceeding_spikes = self.spike_counts >= self.spike_limit
        # if exceeding_spikes.any():
        #     if self.batch_size == 1:
        #         self.thresh = torch.tensor(float('inf'))
        #     else:
        #         print(self.thresh.shape, self.original_thresh, self.thresh)
        #         exceeding_indices = torch.nonzero(self.spike_counts >= self.spike_limit, as_tuple=False)
        #         batch_indices = exceeding_indices[:, 0]  # Get the batch indices
        #         unique_batches = torch.unique(batch_indices)  # Get unique batch indices
        #         # print(unique_batches)
        #         # Set threshold to inf for exceeding batch elements
        #         print(exceeding_indices, batch_indices, unique_batches, self.batch_size)
        #         for batch_idx in unique_batches:
        #             self.thresh[batch_idx] = float('inf')
        
        super().forward(*args, **kwargs)
    def reset_state_variables(self) -> None:
        # Reset state variables including spike count
        super().reset_state_variables()
        self.spike_counts = torch.zeros((self.batch_size,1), device=self.device)
        self.energy_usage = torch.zeros((self.batch_size, self.n))
        # print(self.thresh)
        # self.thresh.fill_(self.original_thresh)
        # print(self.original_thresh)
        # print(self.thresh)

def calculate_intermediate_usefulness(intermediate_spike_record, weight_matrix, correct_neuron_indices, total_time):
    """
    Calculate the usefulness of intermediate layer neurons based on their spike contributions
    to the correct classification neuron in the final layer.

    Parameters:
    - intermediate_spike_record: A tensor of shape [time, batch_size, intermediate_neurons].
    - weight_matrix: A tensor of shape [intermediate_neurons, final_neurons] representing weights from intermediate to final layer.
    - correct_neuron_indices: A tensor of shape [batch_size] with indices of the correct classification neurons for each batch.
    - total_time: Total simulation time.

    Returns:
    - usefulness_scores: A tensor of usefulness scores for each intermediate neuron for each batch.
    """
    # Sum spikes along the time dimension
    spike_sums = intermediate_spike_record.sum(dim=0)  # Shape: [batch_size, intermediate_neurons]

    # Initialize the usefulness scores tensor
    intermediate_neurons = spike_sums.shape[1]
    usefulness_scores = torch.zeros(intermediate_neurons)

    # Iterate over each batch element
    for batch_idx in range(spike_sums.shape[0]):
        correct_neuron_index = correct_neuron_indices[batch_idx]

        # Get the weights for the correct output neuron
        weights = weight_matrix[:, correct_neuron_index]  # Shape: [intermediate_neurons]

        # Update usefulness scores based on weights and spikes
        positive_weights = weights > 0
        negative_weights = weights < 0

        usefulness_scores[positive_weights] += spike_sums[batch_idx, positive_weights]
        usefulness_scores[negative_weights] -= spike_sums[batch_idx, negative_weights]

    return usefulness_scores



percentile = 99.999
random_seed = 0
torch.manual_seed(random_seed)

# batch_size = 32
time = 100

ANN_accuracy = 0
SNN_accuracy = 0

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
    print("Cuda is available")
else:
    device = torch.device('cpu')
    print("Cuda is not available")

class Net(nn.Module):
    def __init__(self, reg_strength=0.01, clip_value=1.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*32*32, num_hidden, bias=False)
        self.fc2 = nn.Linear(num_hidden, 10, bias=False)
        self.clip_value=clip_value


    def forward(self, x):
            x = x.view(-1, 3*32*32)
            x = F.relu(self.fc1(x))
            x = self.fc2(x)
            return F.log_softmax(x)
    def clip_weights(self):
        # Clip the weights of fc1 and fc2 to be within the range [-clip_value, clip_value]
        for layer in [self.fc1, self.fc2]:
            for param in layer.parameters():
                param.data = torch.clamp(param.data, -self.clip_value, self.clip_value)
    def normalize_weights(model):
        with torch.no_grad():
            model.fc1.weight.data /= 2.5
            model.fc2.weight.data /= 2.5

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

class FlattenTransform:
    def __call__(self, x):
        return x.view(-1, 3*32*32)

train_dataset = datasets.CIFAR10('./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),FlattenTransform()]))

train_dataset2 = datasets.CIFAR10('./data',
                               train=False,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),FlattenTransform()]))
print("length of dataset 1 then 2: ", len(train_dataset), len(train_dataset2))
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True, batch_size=100, generator=torch.Generator(device=device))

train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,
                                           shuffle=True, batch_size=batch_size,generator=torch.Generator(device=device))


# for d, target in train_loader:
#     data = d.to(device)

# ANVN = ANVN(2,300)
# print(ANVN.root.children[0].children[0].children)

model = Net()

model.load_state_dict(torch.load("trained_model_cf_256.pt"))
model.normalize_weights()
# model = torch.load('trained_model.pt')

print()
print('Converting ANN to SNN...')

data=None
SNN = ann_to_snn(model, input_shape=(3,32,32), data=data, percentile=percentile, node_type=ANVN_SRIFNodes)

print(SNN)

SNN.add_monitor(
    Monitor(SNN.layers['2'], state_vars=['v'], time=time), name='2'
)

SNN.to(device)
i=0
for connection in SNN.connections.values():
    if "MaxPool2d" not in connection.__class__.__name__:
        print(f"Connection Name: {connection.source, connection.target}")
        print(f"Weight Matrix Shape: {connection.w.shape}")
        if(i==0):
            print(f"Weight matrix: {connection.w}")
        print(f"{connection.type}")
        print("==============================")
        connection.traces=True
    else:
        print("=================\nMaxpool\n==================\n")
        print(f"{connection.type}")
        
    #Batch size weird with multi-layered networks
    i=i+1

correct = 0

voltage_ims = None
voltage_axes= None
ims = [None] * len(set(SNN.layers))
axes = [None] * len(set(SNN.layers))
spike_ims, spike_axes = None, None
spikes = {}
for layer in set(SNN.layers):
    spikes[layer] = Monitor(
        SNN.layers[layer], state_vars=["s"], time=int(time / 1.0), device=device
    )
    SNN.add_monitor(spikes[layer], name="%s_spikes" % layer)

def validate():
    global ANN_accuracy
    model.eval()
    val_loss, correct = 0, 0
    for data, target in train_loader2:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()


    ANN_accuracy = 100. * correct.to(torch.float32) / len(train_loader2.dataset)

    print("ANN accuracy:", ANN_accuracy)
# print(SNN.connections["0","1"].w.shape)
validate()
num_data = 0

# start = t_()
ciu = []
net_spikes = 0

baseline_pos = 0
baseline_neg = 0
baseline_mag = 0
baseline_pv = 0
baseline_nv = 0
baseline_mv = 0

neuron_spikes = None
for conn in set(SNN.connections.values()):
    
    baseline_pos += torch.sum(conn.w[conn.w>0])
    baseline_neg += torch.sum(conn.w[conn.w<0])
    baseline_mag += torch.sum(torch.abs(conn.w))
if os.path.exists("hidden_spikes_256.pt"):
    neuron_spikes = torch.load("hidden_spikes_256.pt")
else:
    for index, (data, target) in enumerate(train_loader):
        # if index * batch_size > 100:
            # break
        start = t_()
        # print(index*batch_size)
        # if index > 100:
        #     break
        num_data +=100
        # print('sample ', index+1, 'elapsed', t_() - start)
        start = t_()
    
        data = data.to(device)
        data = data.view(-1, 3*32*32)
        # print(data.shape)
        inpts = {'Input': data.repeat(time, 1, 1)}
        # print(inpts["Input"].shape)
        # print(inpts["Input"].shape)
        # print(inpts["Input"].shape)
        SNN.run(inputs=inpts, time=time)
        s = {layer: SNN.monitors[f'{layer}_spikes'].get('s') for layer in SNN.layers}
        voltages = {layer: SNN.monitors[layer].get('v') for layer in ['2'] if not layer == 'Input'}
        # pred = torch.argmax(voltages['2'].sum(1))
        # summed_voltages = voltages['2'].sum(0)
        # print(summed_voltages.shape)
        # print(s['2'].shape)
        if neuron_spikes == None:
            neuron_spikes = s['1'].sum((0,1))
        else:
            neuron_spikes += s['1'].sum((0,1))
        summed_spikes=s['2'].sum(0)
        # print(summed_spikes)
        net_spikes += summed_spikes.sum()+ s['1'].sum()
        # pred = torch.argmax(summed_voltages, dim=1).to(device)
        pred = torch.argmax(summed_spikes, dim=1).to(device)
        # print(pred, target)
        # correct += pred.eq(target.data.to(device)).cpu().sum()
        # print(pred)
        # print(target)
        correct += pred.eq(target).sum().item()
        # if index == 0:
        #     ciu = calculate_intermediate_usefulness(s['1'], SNN.connections["1","2"].w, target, time)
        # else:
        #     ciu += calculate_intermediate_usefulness(s['1'], SNN.connections["1","2"].w, target, time)
        spikes_ = {
            layer: spikes[layer].get("s")[:].contiguous() for layer in spikes
        
        }
        # print("Curr time", t_() - start)
        # spikes_ = {
            # layer: spikes2[layer].get("s")[:, 0].contiguous() for layer in spikes2
        # }
        # keys = list(spikes_.keys())
        # for i in range(0, len(keys), 2):
        #     # Get two consecutive layers from spikes_
            
        #     layer1_key = keys[i]
        #     layer2_key = keys[i + 1] if i + 1 < len(keys) else None
            
        #     # Get the spike data for the current layers
        #     layer1_spikes = spikes_[layer1_key]
        #     layer2_spikes = spikes_[layer2_key] if layer2_key else None
        #     if(layer2_spikes == None):
        #         ims[i], axes[i] = plot_spikes(
        #             {layer1_key: layer1_spikes},
        #             ims=ims[i], axes=axes[i]
        #         )
        #     else:
        #         ims[i], axes[i] = plot_spikes(
        #             {layer1_key: layer1_spikes, layer2_key: layer2_spikes},
        #             ims=ims[i], axes=axes[i]
        #         )
        #     for ax in axes[i]:
        #         ax.xaxis.set_major_locator(MultipleLocator(20))
        #         ax.set_xlim(0,100)
        # voltage_ims, voltage_axes = plot_voltages(
        #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
        # )
        SNN.reset_state_variables()
    torch.save(neuron_spikes, "hidden_spikes_256.pt")
# print(neuron_spikes)

import numpy as np
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
SNN.to('cpu')
print("Net spikes: ", net_spikes)

if num_data>0:
    SNN_accuracy = 100.0 * float(correct) / float(num_data)
    print("Net spikes: ", net_spikes/num_data)
    print("accuracy baseline: ", SNN_accuracy)

print("ANVN")
print("="*30)
# neg_indices = ciu < 0
neuron_spikes = neuron_spikes.cpu().numpy()
neuron_spikes = (neuron_spikes-np.min(neuron_spikes))/(np.max(neuron_spikes) - np.min(neuron_spikes))
# neuron_spikes = 2*neuron_spikes - 1

#We want a larger number to mean more energy
#We can convert energy to "lowering threshold" later
# neuron_spikes = - (neuron_spikes-1)
#TRAIN ANVN

# print("Input values:", vascular_tree.root.energy)

# Normalize energy -> bias
# tree_output = 1 - tree_output
# tree_output[tree_output < -1] = -1
# print("Tree output values:", tree_output)

# print("grad:",(neuron_spikes)-tree_output)
energies = [x for x in range(1000,0,-50)]
start=1
multipliers = [1,2,6,8,12,16,20,24,28,30]
results = pd.DataFrame(columns=['Max Energy', 'Energy', 'SNN Accuracy', 'Average Spikes'])
for multiplier in multipliers:
    print("0"*30,"\n\n","0"*30)
    print("multiplier: ", multiplier)
    for energy in energies:
        start-=1
    
        SNN_copy = deepcopy(SNN)
        SNN_copy.to("cuda")
        alg_pos = 0
        alg_neg = 0
        alg_mag = 0
        
        # df = pd.DataFrame({"ANN accuracy":[ANN_accuracy],
        #                    "SNN accuracy": [SNN_accuracy]})
        # ciu = ciu / (max(ciu)*0.9)
        correct2=0
        num_data2 = 0
        net_spikes2 = 0
        ANVN_N = ANVN(2,energy)
        # print(energy)
        for e in range(500):
            tree_output = ANVN_N.root.forward()
            # tree_output = np.maximum(0, tree_output)
            # tree_output = np.minimum(2,tree_output)
            ANVN_N.root.backprop(neuron_spikes)
            # print("ave grad:",np.sum(np.abs(neuron_spikes-(1-tree_output)))/num_hidden)
        print(ANVN_N.root.energy)
        # print("checksum: ", ANVN_N.root.checksum())
        # print( SNN_copy.layers['1'].thresh)
        # print(np.sum(tree_output))
        # - 1*2 = 2
        # 2 - [0,1] = range between 2 and 1
        # multiplier = np.max(tree_output)+1
        # if start==0:
        maxx = multiplier
            # print(maxx)
        greater_mask = tree_output>maxx
        tree_output[greater_mask] = maxx
        # SNN_copy.layers['1'].thresh = SNN_copy.layers['1'].thresh * maxx -torch.tensor(tree_output, device = device)
        tree_output = torch.tensor(tree_output)
        for i,num in enumerate(tree_output):
            SNN_copy.connections["0","1"].w[:,i] = SNN_copy.connections["0","1"].w[:,i] * num
        # print( SNN_copy.layers['1'].thresh)
        for conn in set(SNN_copy.connections.values()):
            alg_pos += torch.sum(conn.w[conn.w>0])
            alg_neg += torch.sum(conn.w[conn.w<0])
            alg_mag += torch.sum(torch.abs(conn.w))
        for index, (data, target) in enumerate(train_loader2):
            # if index > 100:
            #     break
            num_data2 +=batch_size
            # print('sample ', index+1, 'elapsed', t_() - start)
            start = t_()
        
            data = data.to(device)
            data = data.view(-1, 3*32*32)
            inpts = {'Input': data.repeat(time,1, 1)}
            # print(inpts["Input"].shape)
            SNN_copy.run(inputs=inpts, time=time)
            s = {layer: SNN_copy.monitors[f'{layer}_spikes'].get('s') for layer in SNN_copy.layers}
            voltages = {layer: SNN_copy.monitors[layer].get('v') for layer in ['2'] if not layer == 'Input'}
            # pred = torch.argmax(voltages['2'].sum(1))
            # summed_voltages = voltages['2'].sum(0)
            summed_spikes=s['2'].sum(0)
            net_spikes2 += summed_spikes.sum() + s['1'].sum()
            # pred = torch.argmax(summed_voltages, dim=1).to(device)
            pred = torch.argmax(summed_spikes, dim=1).to(device)
            # print(pred, target)
            # correct += pred.eq(target.data.to(device)).cpu().sum()
            correct2 += pred.eq(target).sum().item()
            # print(correct2)
            # accuracy = 100.0 * float(correct) / (index + 1)
            # print(correct)
            # if index == 0:
            #     ciu = calculate_intermediate_usefulness(s['1'], SNN_copy.connections["1","2"].w, target[0], time)
            # else:
            #     ciu += calculate_intermediate_usefulness(s['1'], SNN_copy.connections["1","2"].w, target[0], time)
            # spikes_ = {
            #     layer: spikes[layer].get("s")[:].contiguous() for layer in spikes
            # }
            # spikes_ = {
            #     layer: spikes2[layer].get("s")[:, 0].contiguous() for layer in spikes2
            # }
            # keys = list(spikes_.keys())
            # for i in range(0, len(keys), 2):
            #     # Get two consecutive layers from spikes_
                
            #     layer1_key = keys[i]
            #     layer2_key = keys[i + 1] if i + 1 < len(keys) else None
                
            #     # Get the spike data for the current layers
            #     layer1_spikes = spikes_[layer1_key]
            #     layer2_spikes = spikes_[layer2_key] if layer2_key else None
            #     if(layer2_spikes == None):
            #         ims[i], axes[i] = plot_spikes(
            #             {layer1_key: layer1_spikes},
            #             ims=ims[i], axes=axes[i]
            #         )
            #     else:
            #         ims[i], axes[i] = plot_spikes(
            #             {layer1_key: layer1_spikes, layer2_key: layer2_spikes},
            #             ims=ims[i], axes=axes[i]
            #         )
            #     for ax in axes[i]:
            #         ax.xaxis.set_major_locator(MultipleLocator(20))
            #         ax.set_xlim(0,time)
            # voltage_ims, voltage_axes = plot_voltages(
            #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            # )
            SNN_copy.reset_state_variables()
        print("energy:", energy)
        # SNN_accuracy = 100.0 * float(correct) / float(num_data)
        # print(correct, num_data)
        # print(correct2, num_data2)
        SNN_accuracy2 = 100.0 * float(correct2) / float(10000)
        
        print("accuracy reduced spikes: ", SNN_accuracy2)
        print("net_spikes reduced spikes: ", net_spikes2)
        print("net_spikes reduced spikes #im adjusted: ", net_spikes2/num_data2)
        new_row = pd.DataFrame({
            'Max Energy': [multipliers],
            'Energy': [energy],
            'SNN Accuracy': [SNN_accuracy2],
            'Average Spikes': [(net_spikes2/num_data2).cpu().item()]
        })
        results = pd.concat([results, new_row], ignore_index=True)
        # df.to_csv(f"accuracy_{lam}.csv")
        # print(f"baseline weights pos: {baseline_pos} neg: {baseline_neg} mag {baseline_mag}")
        # print(f"algo weights pos: {alg_pos} neg: {alg_neg} mag {alg_mag}")
        print("="*30)
        del SNN_copy
results.to_excel(f'ANVNSequential_weight.xlsx', index=False)