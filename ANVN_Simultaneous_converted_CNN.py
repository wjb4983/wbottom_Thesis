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
from ANVN_simul import ANVN, ANVN_Node
from ReLU_Scaler import ReLU_Scaler
import numpy as np

batch_size=255
num_hidden=512
max_energy=10
verbose=0
plot=1
loop_max_energy = True
# max_energies = [1,2,4,6,8,10, 12, 14]
# max_energies = [8,10,12,14]
max_energies = [1,2,6,10,14,18,22,26,30,50]
#24 best>?>????


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
            self.s = self.v >= self.thresh_tensor
    
            # Refractoriness and voltage reset
            self.refrac_count.masked_fill_(self.s, self.refrac)
            self.v[self.s] -= self.thresh_tensor[self.s]  # Use correct indexing
    
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
        self.thresh_tensor = self.original_thresh.clone()  # Restore original threshold

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
        # self.thresh_tensor = self.thresh.unsqueeze(0).expand(batch_size, -1)
        self.thresh_tensor= self.thresh
        self.original_thresh = self.thresh.clone()  # Save original thresh
    def set_non_hidden(self):
        self.non_hidden=True
        self.thresh = torch.tensor(1)
        self.thresh_tensor = torch.tensor(1)

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
        # Initialize energy usage and spike counts if necessary
        if self.energy_usage is None or self.energy_usage.size(0) != self.s.shape[0]:
            self.energy_usage = torch.zeros((self.s.shape[0], self.n), device=self.device)
        if self.spike_counts.shape[0] != self.s.shape[0]:
            self.spike_counts = torch.zeros((self.s.shape[0], 1), device=self.device)
    
        if not self.non_hidden:
            self.energy_usage += self.s
            
            # Only update spike counts for batches that haven't already exceeded the spike limit
            self.spike_counts += self.s.sum(dim=1, keepdim=True)
    
            # Lazy initialization of spike_limit_exceeded (only if not already initialized)
            if not hasattr(self, 'spike_limit_exceeded') or self.spike_limit_exceeded.size(0) != self.s.size(0):
                self.spike_limit_exceeded = torch.zeros(self.s.size(0), 1, dtype=torch.bool, device=self.device)
    
            # Update spike limit exceeded status for batches that haven't already exceeded the limit
            if not torch.all(self.spike_limit_exceeded):
                exceeding_spikes = (self.spike_counts >= self.spike_limit) & (~self.spike_limit_exceeded)
    
                # If any batch exceeds the limit
                if exceeding_spikes.any():
                    batch_indices = exceeding_spikes.nonzero(as_tuple=True)[0]
                    self.thresh_tensor[batch_indices, :] = float('inf')
                    self.spike_limit_exceeded[batch_indices] = True  # Mark these batches as having exceeded the limit
        
        super().forward(*args, **kwargs)

    def reset_state_variables(self) -> None:
        # Reset state variables including spike count and threshold
        super().reset_state_variables()
        self.spike_counts = torch.zeros((self.batch_size, 1), device=self.device)
        self.energy_usage = torch.zeros((self.batch_size, self.n), device=self.device)
        self.thresh_tensor = self.original_thresh.clone()  # Restore original threshold
        self.spike_limit_exceeded = torch.zeros(self.s.size(0), 1, dtype=torch.bool, device=self.device)

    def set_spike_limit(self, spike_limit: int) -> None:
        self.spike_limit = spike_limit




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
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) 
        self.conv2 = nn.Conv2d(12, 36, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(2304, num_hidden, bias=False)
        self.dropout = nn.Dropout(0.5)
        self.fc1_ReLU_scaler = ReLU_Scaler(num_hidden)

        self.fc2 = nn.Linear(num_hidden, 10, bias=False)
        self.clip_value=clip_value


    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        intermediate_output = F.relu(self.fc1(x))  # Intermediate output after first layer
        # intermediate_output = self.dropout(intermediate_output)
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

model.load_state_dict(torch.load("cnn_cf_256_test_dropout_sim.pt"))
model.normalize_weights()
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
        data = data.view(-1, 3, 32, 32)
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

SNN.layers['1'].set_non_hidden()
SNN.layers['3'].set_non_hidden()
neuron_spikes = 0

net_spikes
for index, (data, target) in enumerate(train_loader2):
    # if index * batch_size > 100:
        # break
    start = t_()
    # print(index*batch_size)
    # if index > 100:
    #     break
    num_data +=batch_size
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
    voltages = {layer: SNN.monitors[layer].get('v') for layer in ['3'] if not layer == 'Input'}
    # pred = torch.argmax(voltages['2'].sum(1))
    # summed_voltages = voltages['2'].sum(0)
    # print(summed_voltages.shape)
    # print(s['2'].shape)
    neuron_spikes += s['1'].sum((0,1))
    summed_spikes=s['3'].sum(0)
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
    # spikes_ = {
    #     layer: spikes[layer].get("s")[:].contiguous() for layer in spikes
    
    # }
    # # print("Curr time", t_() - start)
    # spikes_ = {
    #     layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
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
    
if num_data>0:
    SNN_accuracy = 100.0 * float(correct) / float(num_data)
    print("Net spikes: ", net_spikes/num_data)
    print("accuracy baseline: ", SNN_accuracy)

neuron_spikes = None
for conn in set(SNN.connections.values()):
    
    baseline_pos += torch.sum(conn.w[conn.w>0])
    baseline_neg += torch.sum(conn.w[conn.w<0])
    baseline_mag += torch.sum(torch.abs(conn.w))
SNN.layers['1'].non_hidden = False
SNN.layers['3'].non_hidden = False
import numpy as np
import copy
from copy import deepcopy
import matplotlib.pyplot as plt
SNN.to('cpu')
print("Net spikes: ", net_spikes)

print("ANVN")
print("="*30)
# neg_indices = ciu < 0
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
# energies = [x for x in range(125,25,-25)]
results = pd.DataFrame(columns=['Max Energy', 'Energy', 'SNN Accuracy', 'Average Spikes'])
import pickle
if loop_max_energy:
    for me in max_energies: 
        print("0"*30)
        print("Max Energy -", me)
        # energies = [0,1,32,128,256,512,1024]
        # energies = [0]
        energies = [x for x in range(250,24,-25)]
        for energy in energies:
            
            SNN_copy = deepcopy(SNN)
            SNN_copy.to("cuda")
            SNN_copy.layers["1"].set_spike_limit(99999999999)
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
                with open('ANVN_cnn.pkl', 'rb') as f:
                    ANVN_N = pickle.load(f)
                ANVN_N.energy = energy
                ANVN_N.root.energy=energy
                tree_output = ANVN_N.root.forward()
                print(np.mean(tree_output), np.median(tree_output), np.std(tree_output))
                # print("checksum: ", ANVN_N.root.checksum())
                # print( SNN_copy.layers['1'].thresh)
                # print(np.sum(tree_output))
                # - 1*2 = 2
                # 2 - [0,1] = range between 2 and 1
                
                multiplier = me#np.max(tree_output)-1
                maxx = multiplier
                greater_mask = tree_output>maxx
                tree_output[greater_mask] = maxx
                # plt.figure()
                # counts, bins = np.histogram(tree_output, 30)
                # plt.stairs(counts, bins)
                SNN_copy.layers['3'].set_non_hidden()
                SNN_copy.layers['1'].thresh = (SNN_copy.layers['1'].thresh * maxx -torch.tensor(tree_output, device = device)).unsqueeze(0).repeat(batch_size,1)
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
            SNN_accuracy2 = 100.0 * float(correct2) / float(10000)
            
            print("accuracy reduced spikes: ", SNN_accuracy2)
            print("net_spikes reduced spikes: ", net_spikes2)
            print("net_spikes reduced spikes #im adjusted: ", net_spikes2/num_data2)
            # df.to_csv(f"accuracy_{lam}.csv")
            # print(f"baseline weights pos: {baseline_pos} neg: {baseline_neg} mag {baseline_mag}")
            # print(f"algo weights pos: {alg_pos} neg: {alg_neg} mag {alg_mag}")
            new_row = pd.DataFrame({
                'Max Energy': [me],
                'Energy': [energy],
                'SNN Accuracy': [SNN_accuracy2],
                'Average Spikes': [(net_spikes2/num_data2).cpu().item()]
            })
            results = pd.concat([results, new_row], ignore_index=True)
            print("="*30)
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
results.to_excel(f'ANVNSimultaneous_CNN.xlsx', index=False)