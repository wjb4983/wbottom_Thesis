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
import types

batch_size=255
num_hidden=512
max_energy=10
verbose=0
plot=0
loop_max_energy = True
max_energies = [6]
nns = [1]#,2,5]

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



import tempfile
from typing import Dict, Iterable, Optional, Type
from bindsnet.learning.reward import AbstractReward
from bindsnet.network.monitors import AbstractMonitor
from bindsnet.network.nodes import CSRMNodes, Nodes
from bindsnet.network.topology import AbstractConnection

def timestep_run(
    self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
) -> None:
    # language=rst
    """
    Simulate network for given inputs and time.

    :param inputs: Dictionary of ``Tensor``s of shape ``[time, *input_shape]`` or
                  ``[time, batch_size, *input_shape]``.
    :param time: Simulation time.
    :param one_step: Whether to run the network in "feed-forward" mode, where inputs
        propagate all the way through the network in a single simulation time step.
        Layers are updated in the order they are added to the network.

    Keyword arguments:

    :param Dict[str, torch.Tensor] clamp: Mapping of layer names to boolean masks if
        neurons should be clamped to spiking. The ``Tensor``s have shape
        ``[n_neurons]`` or ``[time, n_neurons]``.
    :param Dict[str, torch.Tensor] unclamp: Mapping of layer names to boolean masks
        if neurons should be clamped to not spiking. The ``Tensor``s should have
        shape ``[n_neurons]`` or ``[time, n_neurons]``.
    :param Dict[str, torch.Tensor] injects_v: Mapping of layer names to boolean
        masks if neurons should be added voltage. The ``Tensor``s should have shape
        ``[n_neurons]`` or ``[time, n_neurons]``.
    :param Union[float, torch.Tensor] reward: Scalar value used in reward-modulated
        learning.
    :param Dict[Tuple[str], torch.Tensor] masks: Mapping of connection names to
        boolean masks determining which weights to clamp to zero.
    :param Bool progress_bar: Show a progress bar while running the network.

    **Example:**

    .. code-block:: python

        import torch
        import matplotlib.pyplot as plt

        from bindsnet.network import Network
        from bindsnet.network.nodes import Input
        from bindsnet.network.monitors import Monitor

        # Build simple network.
        network = Network()
        network.add_layer(Input(500), name='I')
        network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')

        # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
        spikes = torch.bernoulli(0.5 * torch.rand(500, 500))

        # Run network simulation.
        network.run(inputs={'I' : spikes}, time=500)

        # Look at input spiking activity.
        spikes = network.monitors['I'].get('s')
        plt.matshow(spikes, cmap='binary')
        plt.xticks(()); plt.yticks(());
        plt.xlabel('Time'); plt.ylabel('Neuron index')
        plt.title('Input spiking')
        plt.show()
    """
    # Check input type
    assert type(inputs) == dict, (
        "'inputs' must be a dict of names of layers "
        + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
    )
    # Parse keyword arguments.
    clamps = kwargs.get("clamp", {})
    unclamps = kwargs.get("unclamp", {})
    masks = kwargs.get("masks", {})
    injects_v = kwargs.get("injects_v", {})

    # Compute reward.
    if self.reward_fn is not None:
        kwargs["reward"] = self.reward_fn.compute(**kwargs)

    # Dynamic setting of batch size.
    if inputs != {}:
        for key in inputs:
            # goal shape is [time, batch, n_0, ...]
            if len(inputs[key].size()) == 1:
                # current shape is [n_0, ...]
                # unsqueeze twice to make [1, 1, n_0, ...]
                inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
            elif len(inputs[key].size()) == 2:
                # current shape is [time, n_0, ...]
                # unsqueeze dim 1 so that we have
                # [time, 1, n_0, ...]
                inputs[key] = inputs[key].unsqueeze(1)

        for key in inputs:
            # batch dimension is 1, grab this and use for batch size
            if inputs[key].size(1) != self.batch_size:
                self.batch_size = inputs[key].size(1)

                for l in self.layers:
                    self.layers[l].set_batch_size(self.batch_size)

                for m in self.monitors:
                    self.monitors[m].reset_state_variables()

            break

    # Effective number of timesteps.
    timesteps = int(time / self.dt)

    # Run synapse updates.
    if "a_minus" in kwargs:
        A_Minus = kwargs["a_minus"]
        kwargs.pop("a_minus")
        if isinstance(A_Minus, dict):
            A_MD = True
        else:
            A_MD = False
    else:
        A_Minus = None

    if "a_plus" in kwargs:
        A_Plus = kwargs["a_plus"]
        kwargs.pop("a_plus")
        if isinstance(A_Plus, dict):
            A_PD = True
        else:
            A_PD = False
    else:
        A_Plus = None

    # Simulate network activity for `time` timesteps.
    for t in range(timesteps):
        # Get input to all layers (synchronous mode).
        current_inputs = {}
        if not one_step:
            current_inputs.update(self._get_inputs())

        for l in self.layers:
            # Update each layer of nodes.
            if l in inputs:
                if l in current_inputs:
                    current_inputs[l] += inputs[l][t]
                else:
                    current_inputs[l] = inputs[l][t]

            if one_step:
                # Get input to this layer (one-step mode).
                current_inputs.update(self._get_inputs(layers=[l]))

            # Inject voltage to neurons.
            inject_v = injects_v.get(l, None)
            if inject_v is not None:
                if inject_v.ndimension() == 1:
                    self.layers[l].v += inject_v
                else:
                    self.layers[l].v += inject_v[t]
            if l in current_inputs:
                if l=='Input':
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(x=current_inputs[l], timestep=t)
            else:
                self.layers[l].forward(
                    x=torch.zeros(
                        self.layers[l].s.shape, device=self.layers[l].s.device
                    )
                )

            # Clamp neurons to spike.
            clamp = clamps.get(l, None)
            if clamp is not None:
                if clamp.ndimension() == 1:
                    self.layers[l].s[:, clamp] = 1
                else:
                    self.layers[l].s[:, clamp[t]] = 1

            # Clamp neurons not to spike.
            unclamp = unclamps.get(l, None)
            if unclamp is not None:
                if unclamp.ndimension() == 1:
                    self.layers[l].s[:, unclamp] = 0
                else:
                    self.layers[l].s[:, unclamp[t]] = 0

        for c in self.connections:
            flad_m = False
            if A_Minus != None and ((isinstance(A_Minus, float)) or (c in A_Minus)):
                if A_MD:
                    kwargs["a_minus"] = A_Minus[c]
                else:
                    kwargs["a_minus"] = A_Minus
                flad_m = True

            flad_p = False
            if A_Plus != None and ((isinstance(A_Plus, float)) or (c in A_Plus)):
                if A_PD:
                    kwargs["a_plus"] = A_Plus[c]
                else:
                    kwargs["a_plus"] = A_Plus
                flad_p = True

            self.connections[c].update(
                mask=masks.get(c, None), learning=self.learning, **kwargs
            )
            if flad_m:
                kwargs.pop("a_minus")
            if flad_p:
                kwargs.pop("a_plus")

        # # Get input to all layers.
        # current_inputs.update(self._get_inputs())

        # Record state variables of interest.
        for m in self.monitors:
            self.monitors[m].record()

    # Re-normalize connections.
    for c in self.connections:
        self.connections[c].normalize()
    


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

        self.register_buffer("reset", torch.tensor(reset, dtype=torch.float))
        self.register_buffer("thresh", torch.tensor(thresh, dtype=torch.float))
        self.register_buffer("refrac", torch.tensor(refrac, dtype=torch.float))
        self.register_buffer("v", torch.FloatTensor())
        self.register_buffer("refrac_count", torch.FloatTensor())

        self.lbound = lbound
        self.original_thresh = None  # Store original threshold
        self.change = 1
        self.available_energies = torch.zeros(batch_size, num_hidden, dtype=torch.float, device='cuda')
        self.energy_tensor = None

    def forward(self, x: torch.Tensor) -> None:
        """
        Runs a single simulation step.
        :param x: Inputs to the layer.
        """
        if self.non_hidden:
            # language=rst
            """
            Runs a single simulation step.

            :param x: Inputs to the layer.
            """
            self.thresh = torch.tensor(1, dtype=torch.float)
            # Integrate input voltages.
            self.v += (self.refrac_count == 0).float() * x

            # Decrement refractory counters.
            self.refrac_count = (self.refrac_count > 0).float() * (
                self.refrac_count - self.dt
            )

            # Check for spiking neurons.
            self.s = self.v >= self.thresh

            # Refractoriness and voltage reset.
            self.refrac_count.masked_fill_(self.s, self.refrac)
            self.v[self.s] = self.v[self.s] - self.thresh

            # Voltage clipping to lower bound.
            if self.lbound is not None:
                self.v.masked_fill_(self.v < self.lbound, self.lbound)
        else:
            if self.v.dim() != x.dim():
                raise ValueError("Input dimensions must match the neuron state dimensions")
    
            # Integrate input voltages
            self.v += (self.refrac_count == 0).float() * x
    
            # Decrement refractory counters
            self.refrac_count = (self.refrac_count > 0).float() * (self.refrac_count - self.dt)
    
            # Check for spiking neurons
            if self.v.shape[0] < self.thresh_tensor.shape[0]:
                self.thresh_tensor = self.thresh_tensor[:self.v.shape[0],:]
            
            if self.v.shape[0] == self.available_energies.shape[0]:
                enough_energy_mask = self.available_energies >=1
            else:
                self.available_energies = self.available_energies[:self.v.shape[0],:]
                enough_energy_mask = self.available_energies >=1
            
            self.s = (self.v >= self.thresh_tensor) & enough_energy_mask
            self.available_energies[self.s] -=1
    
            # Refractoriness and voltage reset
            self.refrac_count.masked_fill_(self.s, self.refrac)
            self.v[self.s] -= self.thresh_tensor[self.s]  # Use correct indexing
            #THRESHCHANGE
            # self.v[self.s] -= self.thresh_tensor
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
        self.thresh_tensor = self.original_thresh.clone()  # Restore original  #THRESHCHANGE
        # self.thresh_tensor = torch.tensor(1)
        self.available_energies = torch.zeros(batch_size, num_hidden, dtype=torch.float, device='cuda')

    def set_batch_size(self, batch_size: int) -> None:
        """
        Sets mini-batch size.
        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)
        device = self.reset.device

        # Ensure self.thresh is initialized with the correct size
        if self.thresh.dim() == 0:
            self.thresh = self.thresh.expand(self.n)
        elif self.thresh.dim() == 1 and self.thresh.size(0) != self.n:
            raise ValueError(f"Expected threshold tensor of size {self.n}, but got {self.thresh.size(0)}")

        # Initialize voltages and refractory counters
        self.v = self.reset * torch.ones(batch_size, *self.shape, device=device)
        self.refrac_count = torch.zeros_like(self.v, device=device)
        # Create the threshold tensor of shape [batch_size, n_neurons]
        #XXXXXX# self.thresh_tensor = self.thresh.unsqueeze(0).expand(batch_size, -1)
        self.thresh_tensor= self.thresh #THRESHCHANGE
        # self.thresh_tensor = torch.tensor(1)
        self.original_thresh = self.thresh.clone()  # Save original thresh
    def set_non_hidden(self):
        self.non_hidden=True
        self.thresh = torch.tensor(1)
        self.thresh_tensor = torch.tensor(1)
    def set_energy_tensor(self, energy_tensor):
        self.energy_tensor = energy_tensor



class ANVN_SRIFNodes(SubtractiveResetIFNodes):
    def __init__(self, *args, spike_limit=1000, device='cuda', batch_size=batch_size, **kwargs):
        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device
        self.spike_counts = torch.zeros((self.batch_size, 1), device=self.device)
        self.spike_limit = spike_limit
        self.energy_usage = torch.zeros((self.batch_size, self.n), device=self.device)
        self.done = False
        self.non_hidden=False
        self.ANVN_interval = 0

    # def forward(self, *args, **kwargs):
    #     # Increment spike count for each batch element
    #     if self.energy_usage is None or self.energy_usage.size(0) != self.s.shape[0]:
    #         self.energy_usage = torch.zeros((self.s.shape[0], self.n), device=self.device)
    #     if self.spike_counts.shape[0] != self.s.shape[0]:
    #         self.spike_counts = torch.zeros((self.s.shape[0], 1), device=self.device)
            
    #     self.energy_usage += self.s
    #     if not self.non_hidden:
    #         self.spike_counts += self.s.sum(dim=1, keepdim=True)
            
    #         # Check if spike limit reached for any batch element
    #         exceeding_spikes = self.spike_counts >= self.spike_limit
            
    #         if exceeding_spikes.any():
    #             # Set threshold to inf for exceeding batch elements
    #             batch_indices = exceeding_spikes.nonzero(as_tuple=True)[0]
    #             self.thresh_tensor[batch_indices, :] = float('inf')
        
    #     super().forward(*args, **kwargs)
    def forward(self, *args, **kwargs):
        timestep = kwargs.pop('timestep', None)
        # Initialize energy usage and spike counts if necessary
        if self.energy_usage is None or self.energy_usage.size(0) != self.s.shape[0]:
            self.energy_usage = torch.zeros((self.s.shape[0], self.n), device=self.device)
        if self.spike_counts.shape[0] != self.s.shape[0]:
            self.spike_counts = torch.zeros((self.s.shape[0], 1), device=self.device)
        if not self.non_hidden:
            if timestep % self.ANVN_interval == 0:
                self.available_energies += self.energy_tensor
                
            self.energy_usage += self.s
            
            # # Only update spike counts for batches that haven't already exceeded the spike limit
            # self.spike_counts += self.s.sum(dim=1, keepdim=True)
    
            # # Lazy initialization of spike_limit_exceeded (only if not already initialized)
            # if not hasattr(self, 'spike_limit_exceeded') or self.spike_limit_exceeded.size(0) != self.s.size(0):
            #     self.spike_limit_exceeded = torch.zeros(self.s.size(0), 1, dtype=torch.bool, device=self.device)
    
            # # Update spike limit exceeded status for batches that haven't already exceeded the limit
            # if not torch.all(self.spike_limit_exceeded):
            #     exceeding_spikes = (self.spike_counts >= self.spike_limit) & (~self.spike_limit_exceeded)
    
            #     # If any batch exceeds the limit
            #     if exceeding_spikes.any():
            #         batch_indices = exceeding_spikes.nonzero(as_tuple=True)[0]
            #         self.thresh_tensor[batch_indices, :] = float('inf')
            #         self.spike_limit_exceeded[batch_indices] = True  # Mark these batches as having exceeded the limit
        super().forward(*args, **kwargs)

    def reset_state_variables(self) -> None:
        # Reset state variables including spike count and threshold
        super().reset_state_variables()
        self.spike_counts = torch.zeros((self.batch_size, 1), device=self.device)
        self.energy_usage = torch.zeros((self.batch_size, self.n), device=self.device)
        # self.thresh_tensor = self.original_thresh.clone()  # Restore original threshold #THRESHCHANGE
        self.spike_limit_exceeded = torch.zeros(self.s.size(0), 1, dtype=torch.bool, device=self.device)

    def set_spike_limit(self, spike_limit: int) -> None:
        self.spike_limit = spike_limit

time = 100

percentile = 99.999
random_seed = 0
torch.manual_seed(random_seed)

# batch_size = 32
# time = 100

ANN_accuracy = 0
SNN_accuracy = 0

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
    print("Cuda is available")
else:
    device = torch.device('cpu')
    print("Cuda is not available")

from ANVN_simul import ANVN, ANVN_Node
from ReLU_Scaler import ReLU_Scaler

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
    def normalize_weights(model):
        with torch.no_grad():
            model.fc1.weight.data /= 2.5
            model.fc2.weight.data /= 2.5

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

model.load_state_dict(torch.load("trained_model_cf_256_test!!!.pt"))
# model.normalize_weights()
# model = torch.load('trained_model.pt')

print()
print('Converting ANN to SNN...')

data=None
from bindsnet.network.nodes import LIFNodes
SNN = ann_to_snn(model, input_shape=(3,32,32), data=data, percentile=percentile, node_type=ANVN_SRIFNodes)#, node_type=LIFNodes)# node_type=ANVN_SRIFNodes)
# SNN.connections['0','1'].w *=7
# SNN.connections['2','3'].w *=100
# SNN.layers['1'].refrac=torch.tensor(12)
# SNN.layers['3'].refrac=torch.tensor(5)
# SNN.layers['1'].tc_decay=torch.tensor(20)
# SNN.layers['3'].tc_decay=torch.tensor(10)
SNN.run = types.MethodType(timestep_run, SNN)

print(SNN)

SNN.add_monitor(
    Monitor(SNN.layers['3'], state_vars=['v'], time=time), name='3'
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
        output, _ = model(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()


    ANN_accuracy = 100. * correct.to(torch.float32) / len(train_loader2.dataset)

    print("ANN accuracy:", ANN_accuracy)
print(SNN.connections["0","1"].w.shape)
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


neuron_spikes = 0

net_spikes

if num_data>0:
    SNN_accuracy = 100.0 * float(correct) / float(num_data)
    print("Net spikes: ", net_spikes/num_data)
    print("accuracy baseline: ", SNN_accuracy)

neuron_spikes = None
for conn in set(SNN.connections.values()):
    
    baseline_pos += torch.sum(conn.w[conn.w>0])
    baseline_neg += torch.sum(conn.w[conn.w<0])
    baseline_mag += torch.sum(torch.abs(conn.w))

import numpy as np
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
SNN.to('cpu')
print("Net spikes: ", net_spikes)

print("ANVN")
print("="*30)

<<<<<<< HEAD
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
# energies = [x for x in range(125,25,-25)]
times = [10]
=======
times = [20]
>>>>>>> a1b5553f2ea75b6dd33dfebb5f391d9625543a45
for time in times:
    SNN.add_monitor(
        Monitor(SNN.layers['3'], state_vars=['v'], time=time), name='3'
    )
    spikes = {}
    for layer in set(SNN.layers):
        spikes[layer] = Monitor(
            SNN.layers[layer], state_vars=["s"], time=int(time / 1.0), device=device
        )
        SNN.add_monitor(spikes[layer], name="%s_spikes" % layer)
    import pickle
    for n in nns:
        results = pd.DataFrame(columns=['Max Energy', 'Energy','Refill Energy', 'SNN Accuracy', 'Average Spikes'])
        
        if loop_max_energy:
            for me in max_energies: 
                print("0"*30)
                print("Max Energy -", me)
                # energies = [0,1,32,128,256,512,1024]
                # energies = [0]
                energies = [x for x in range(50, 800, 100)]
                refill_energies = [x for x in range(100, 1000, 100)]
                for energy in energies:
                    for re in refill_energies:
                        SNN_copy = deepcopy(SNN)
                        SNN_copy.to("cuda")
                        # SNN_copy.layers["1"].set_spike_limit(9999999999)
                        # print(SNN_copy.layers["1"].spike_limit)
                        alg_pos = 0
                        alg_neg = 0
                        alg_mag = 0
                        
                        # df = pd.DataFrame({"ANN accuracy":[ANN_accuracy],
                        #                    "SNN accuracy": [SNN_accuracy]})
                        # ciu = ciu / (max(ciu)*0.9)
                        correct2=0
                        num_data2 = 0
                        net_spikes2 = 0
                        if energy != 0:
                            ANVN_N = ANVN(2,energy)
                            ANVN_N.root.clip()
                            with open('ANVN_updated.pkl', 'rb') as f:
                                ANVN_N = pickle.load(f)
            
                            ANVN_N.energy = energy
                            ANVN_N.root.energy=energy
                            tree_output = ANVN_N.root.forward()
                            # tree_output2 = torch.full_like(torch.tensor(tree_output), 99999999)
                            
                            SNN_copy.layers['1'].ANVN_interval = n
                            SNN_copy.layers['3'].ANVN_interval = n
            
                            print(np.mean(tree_output), np.median(tree_output), np.std(tree_output))
                            # print("checksum: ", ANVN_N.root.checksum())
                            # print( SNN_copy.layers['1'].thresh)
                            # print(np.sum(tree_output))
                            
                            multiplier = me#np.max(tree_output)-1
                            maxx = multiplier
                            greater_mask = tree_output>maxx
                            tree_output[greater_mask] = maxx
                            # plt.figure()
                            # counts, bins = np.histogram(tree_output, 30)
                            # plt.stairs(counts, bins)
                            SNN_copy.layers['3'].set_non_hidden()
                            SNN_copy.layers['1'].thresh = (SNN_copy.layers['1'].thresh * maxx -torch.tensor(tree_output, device = device)).unsqueeze(0).repeat(batch_size,1)
                            ANVN_N.energy = re
                            ANVN_N.root.energy=re
                            tree_output2 = ANVN_N.root.forward()
                            # tree_output2 = torch.full_like(torch.tensor(tree_output), 99999999)
                            SNN_copy.layers['1'].set_energy_tensor(torch.tensor(tree_output2))
            
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
                            print(data.shape)
                            data = data.to(device)
                            data = data.view(-1, 3*32*32)
                            inpts = {'Input': data.repeat(time,1, 1)}
                            print(inpts["Input"].shape)
                            SNN_copy.run(inputs=inpts, time=time)
                            s = {layer: SNN_copy.monitors[f'{layer}_spikes'].get('s') for layer in SNN_copy.layers}
                            voltages = {layer: SNN_copy.monitors[layer].get('v') for layer in ['3'] if not layer == 'Input'}
                            # pred = torch.argmax(voltages['2'].sum(1))
                            # summed_voltages = voltages['2'].sum(0)
                            summed_spikes=s['3'].sum(0)
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
                            
                            
                            if plot:
                                keys = list(s.keys())
                                s = {layer: s[layer][:,0,:] for layer in keys}
                                for i in range(0, len(keys), 2):
                                    # Get two consecutive layers from spikes_
                                    
                                    layer1_key = keys[i]
                                    layer2_key = keys[i + 1] if i + 1 < len(keys) else None
                                    
                                    # Get the spike data for the current layers
                                    layer1_spikes = s[layer1_key]
                                    layer2_spikes = s[layer2_key] if layer2_key else None
                                    if(layer2_spikes == None):
                                        ims[i], axes[i] = plot_spikes(
                                            {layer1_key: layer1_spikes},
                                            ims=ims[i], axes=axes[i]
                                        )
                                    else:
                                        ims[i], axes[i] = plot_spikes(
                                            {layer1_key: layer1_spikes, layer2_key: layer2_spikes},
                                            ims=ims[i], axes=axes[i]
                                        )
                                    for ax in axes[i]:
                                        ax.xaxis.set_major_locator(MultipleLocator(20))
                                        ax.set_xlim(0,time)
                                voltage_ims, voltage_axes = plot_voltages(
                                    voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
                                )
                            SNN_copy.reset_state_variables()
                        print("energy:", energy)
                        # SNN_accuracy = 100.0 * float(correct) / float(num_data)
                        # print(correct, num_data)
                        # print(correct2, num_data2)
                        SNN_accuracy2 = 100.0 * float(correct2) / float(10000)
                        
                        print("accuracy reduced spikes: ", SNN_accuracy2)
                        print("net_spikes reduced spikes: ", net_spikes2)
                        print("net_spikes reduced spikes #im adjusted: ", net_spikes2/num_data2)
                        # df.to_csv(f"accuracy_{lam}.csv")
                        # print(f"baseline weights pos: {baseline_pos} neg: {baseline_neg} mag {baseline_mag}")
                        # print(f"algo weights pos: {alg_pos} neg: {alg_neg} mag {alg_mag}")
                        print("="*30)
                        new_row = pd.DataFrame({
                            'Max Energy': [me],
                            'Energy': [energy],
                            'Refill Energy': [re],
                            'SNN Accuracy': [SNN_accuracy2],
                            'Average Spikes': [(net_spikes2/num_data2).cpu().item()]
                        })
                        results = pd.concat([results, new_row], ignore_index=True)
                        del SNN_copy
        else:
            # energies = [0,1,32,128,256,512,1024]
            # energies = [0]
            energies = [x for x in range(250,-1,-25)]
            for energy in energies:
                
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
                if energy != 0:
                    ANVN_N = ANVN(2,energy)
                    ANVN_N.root.clip()
                    with open('ANVN.pkl', 'wb') as f:
                        pickle.dump(ANVN_N, f)
                    ANVN_N.energy = energy
                    ANVN_N.root.energy=energy
                    tree_output = ANVN_N.root.forward()
                    print(np.mean(tree_output), np.median(tree_output), np.std(tree_output))
                    # print("checksum: ", ANVN_N.root.checksum())
                    # print( SNN_copy.layers['1'].thresh)
                    # print(np.sum(tree_output))
                    # - 1*2 = 2
                    # 2 - [0,1] = range between 2 and 1
                    
                    multiplier = max_energy#np.max(tree_output)-1
                    maxx = multiplier
                    greater_mask = tree_output>maxx
                    tree_output[greater_mask] = maxx
                    plt.figure()
                    counts, bins = np.histogram(tree_output, 30)
                    plt.stairs(counts, bins)
                    # SNN_copy.layers['1'].thresh = (SNN_copy.layers['1'].thresh * maxx -torch.tensor(tree_output, device = device)).unsqueeze(0).repeat(batch_size,1)
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
                    voltages = {layer: SNN_copy.monitors[layer].get('v') for layer in ['3'] if not layer == 'Input'}
                    # pred = torch.argmax(voltages['2'].sum(1))
                    # summed_voltages = voltages['2'].sum(0)
                    summed_spikes=s['3'].sum(0)
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
                    
                    
                    if plot:
                        keys = list(s.keys())
                        s = {layer: s[layer][:,0,:] for layer in keys}
                        for i in range(0, len(keys), 2):
                            # Get two consecutive layers from spikes_
                            
                            layer1_key = keys[i]
                            layer2_key = keys[i + 1] if i + 1 < len(keys) else None
                            
                            # Get the spike data for the current layers
                            layer1_spikes = s[layer1_key]
                            layer2_spikes = s[layer2_key] if layer2_key else None
                            if(layer2_spikes == None):
                                ims[i], axes[i] = plot_spikes(
                                    {layer1_key: layer1_spikes},
                                    ims=ims[i], axes=axes[i]
                                )
                            else:
                                ims[i], axes[i] = plot_spikes(
                                    {layer1_key: layer1_spikes, layer2_key: layer2_spikes},
                                    ims=ims[i], axes=axes[i]
                                )
                            for ax in axes[i]:
                                ax.xaxis.set_major_locator(MultipleLocator(20))
                                ax.set_xlim(0,time)
                        voltage_ims, voltage_axes = plot_voltages(
                            voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
                        )
                    SNN_copy.reset_state_variables()
                print("energy:", energy)
                # SNN_accuracy = 100.0 * float(correct) / float(num_data)
                # print(correct, num_data)
                # print(correct2, num_data2)
                SNN_accuracy2 = 100.0 * float(correct2) / float(num_data2)
                
                print("accuracy reduced spikes: ", SNN_accuracy2)
                print("net_spikes reduced spikes: ", net_spikes2)
                print("net_spikes reduced spikes #im adjusted: ", net_spikes2/num_data2)
                # df.to_csv(f"accuracy_{lam}.csv")
                # print(f"baseline weights pos: {baseline_pos} neg: {baseline_neg} mag {baseline_mag}")
                # print(f"algo weights pos: {alg_pos} neg: {alg_neg} mag {alg_mag}")
                print("="*30)
                del SNN_copy
                
        results.to_excel(f'ANVNSimultaneous_{n}n_root_bank_varyenergy_updated_{time}time.xlsx', index=False)