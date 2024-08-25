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

class SubtractiveResetIFNodes(nodes.Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <https://bit.ly/2EOk6YN>` using
    reset by subtraction.
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
        lbound: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons with the subtractive reset mechanism
        from `this paper <https://bit.ly/2ShuwrQ>`_.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param traces_additive: Whether to record spike traces additively.
        :param tc_trace: Time constant of spike trace decay.
        :param trace_scale: Scaling factor for spike trace.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        :param lbound: Lower bound of the voltage.
        """
        super().__init__(
            n=n,
            shape=shape,
            traces=traces,
            traces_additive=traces_additive,
            tc_trace=tc_trace,
            trace_scale=trace_scale,
            sum_input=sum_input,
        )

        self.register_buffer(
            "reset", torch.tensor(reset, dtype=torch.float)
        )  # Post-spike reset voltage.
        self.register_buffer(
            "thresh", torch.tensor(thresh, dtype=torch.float)
        )  # Spike threshold voltage.
        self.register_buffer(
            "refrac", torch.tensor(refrac)
        )  # Post-spike refractory period.
        self.register_buffer("v", torch.FloatTensor())  # Neuron voltages.
        self.register_buffer(
            "refrac_count", torch.FloatTensor()
        )  # Refractory period counters.

        self.lbound = lbound  # Lower bound of voltage.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param x: Inputs to the layer.
        """
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

        super().forward(x)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_state_variables()
        self.v.fill_(self.reset)  # Neuron voltages.
        self.refrac_count.zero_()  # Refractory period counters.

    def set_batch_size(self, batch_size) -> None:
        # language=rst
        """
        Sets mini-batch size. Called when layer is added to a network.

        :param batch_size: Mini-batch size.
        """
        super().set_batch_size(batch_size=batch_size)

        device = self.reset.device
        # shape_tensor = torch.tensor(self.shape, device=device)
        self.v.to(device)
        print((self.v.device), (self.reset.device), (self.shape))
        self.v = self.reset * torch.ones(batch_size, *self.shape, device=self.reset.device)
        
        self.refrac_count = torch.zeros_like(self.v, device=self.refrac_count.device)
class Custom_SRIFNodes(SubtractiveResetIFNodes):
    def __init__(self, *args, spike_limit=1000, device = 'cuda', batch_size=1, **kwargs):

        super().__init__(*args, **kwargs)
        self.batch_size = batch_size
        self.device = device
        self.spike_counts = torch.zeros((self.batch_size,1), device=self.device)
        self.spike_limit = spike_limit
        self.original_thresh = self.thresh
        

    def forward(self, *args, **kwargs):
        # Increment spike count for each batch element
        self.spike_counts += self.s.sum(dim=1, keepdim=True)
        # print(self.s.sum(dim=1, keepdim=True), self.spike_counts)
        # print(torch.unique(self.thresh))
        # Check if spike limit reached for any batch element
        exceeding_spikes = self.spike_counts >= self.spike_limit
        if exceeding_spikes.any():
            if self.batch_size == 1:
                self.thresh = torch.tensor(float('inf'))
            else:
                exceeding_indices = torch.nonzero(self.spike_counts >= self.spike_limit, as_tuple=False)
                batch_indices = exceeding_indices[:, 0]  # Get the batch indices
                unique_batches = torch.unique(batch_indices)  # Get unique batch indices
                # print(unique_batches)
                # Set threshold to inf for exceeding batch elements
                print(exceeding_indices, batch_indices, unique_batches, self.batch_size)
                for batch_idx in unique_batches:
                    self.thresh[batch_idx] = float('inf')
        
        super().forward(*args, **kwargs)
    def reset_state_variables(self) -> None:
        # Reset state variables including spike count
        super().reset_state_variables()
        self.spike_counts = torch.zeros((self.batch_size,1), device=self.device)
        # print(self.thresh)
        self.thresh.fill_(self.original_thresh)
        # print(self.original_thresh)
        # print(self.thresh)


def calculate_intermediate_usefulness(intermediate_spike_record, weight_matrix, correct_neuron_index, total_time):
    """
    Calculate the usefulness of intermediate layer neurons based on their spike contributions
    to the correct classification neuron in the final layer.

    Parameters:
    - intermediate_spike_record: A tensor of shape [time, 1, intermediate_neurons].
    - weight_matrix: A tensor of shape [intermediate_neurons, final_neurons] representing weights from intermediate to final layer.
    - correct_neuron_index: Index of the correct classification neuron in the final layer.
    - total_time: Total simulation time.

    Returns:
    - usefulness_scores: A tensor of usefulness scores for each intermediate neuron.
    """
    # intermediate_neurons = intermediate_spike_record.shape[2]
    # usefulness_scores = torch.zeros(intermediate_neurons)

    # for t in range(total_time):
    #     for neuron in range(intermediate_neurons):
    #         spike_value = intermediate_spike_record[t, 0, neuron].item()  # Convert to scalar
    #         if spike_value > 0:
    #             weight = weight_matrix[neuron, correct_neuron_index].item()  # Convert to scalar
    #             if weight > 0:
    #                 usefulness_scores[neuron] += 1
    #             elif weight < 0:
    #                 usefulness_scores[neuron] -= 1
    # intermediate_neurons = intermediate_spike_record.shape[2]
    intermediate_neurons = intermediate_spike_record.shape[2]
    
    # Create a boolean mask for spikes and another for positive weights
    spikes = intermediate_spike_record[:, 0, :] > 0
    weights = weight_matrix[:, correct_neuron_index]
    
    # Create a mask for positive and negative weights
    positive_weights = weights > 0
    negative_weights = weights < 0
    
    # Initialize the usefulness scores to zero
    usefulness_scores = torch.zeros(intermediate_neurons, device=intermediate_spike_record.device)
    
    # Use torch.where to conditionally update usefulness_scores
    usefulness_scores += torch.sum(spikes & positive_weights, dim=0)
    usefulness_scores -= torch.sum(spikes & negative_weights, dim=0)
    
    # # Verification
    # usefulness_scores2 = torch.zeros(intermediate_neurons, device=intermediate_spike_record.device)
    # for t in range(total_time):
    #     for neuron in range(intermediate_neurons):
    #         spike_value = intermediate_spike_record[t, 0, neuron].item()  # Convert to scalar
    #         if spike_value > 0:
    #             weight = weight_matrix[neuron, correct_neuron_index].item()  # Convert to scalar
    #             if weight > 0:
    #                 usefulness_scores2[neuron] += 1
    #             elif weight < 0:
    #                 usefulness_scores2[neuron] -= 1
    
    # # Check if both methods give the same results
    # print((usefulness_scores == usefulness_scores2).all())

    return usefulness_scores


percentile = 99.9
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
        self.fc1 = nn.Linear(3*32*32, 300, bias=False)
        self.fc2 = nn.Linear(300, 10, bias=False)
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

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True, batch_size=train_dataset.__len__(), generator=torch.Generator(device=device))

train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,
                                           shuffle=True, generator=torch.Generator(device=device))


# for d, target in train_loader:
#     data = d.to(device)



model = Net()

model.load_state_dict(torch.load("trained_model_cf.pt"))
model.normalize_weights()
# model = torch.load('trained_model.pt')

print()
print('Converting ANN to SNN...')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data=None
SNN = ann_to_snn(model, input_shape=(3,32,32), data=data, percentile=percentile, node_type=Custom_SRIFNodes)
SNN.to(device)
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
print(SNN.connections["0","1"].w.shape)
validate()
num_data = 0

start = t_()
ciu = []
net_spikes = 0
baseline_pos = 0
baseline_neg = 0
baseline_mag = 0
for conn in set(SNN.connections.values()):
    
    baseline_pos += torch.sum(conn.w[conn.w>0])
    baseline_neg += torch.sum(conn.w[conn.w<0])
    baseline_mag += torch.sum(torch.abs(conn.w))
print("Normal")
for index, (data, target) in enumerate(train_loader2):
    if index > 100:
        break
    num_data +=1
    print('sample ', index+1, 'elapsed', t_() - start)
    start = t_()

    data = data.to(device)
    data = data.view(-1, 3*32*32)
    inpts = {'Input': data.repeat(time, 1)}
    # if device:
    #     inpts = {k: v.cuda() for k, v in inpts.items()}
    # print(inpts["Input"].shape)
    SNN.run(inputs=inpts, time=time)
    s = {layer: SNN.monitors[f'{layer}_spikes'].get('s') for layer in SNN.layers}
    # voltages = {layer: SNN.monitors[layer].get('v') for layer in ['2'] if not layer == 'Input'}
    # pred = torch.argmax(voltages['2'].sum(1))
    # summed_voltages = voltages['2'].sum(0)
    summed_spikes=s['2'].sum(0)
    net_spikes += summed_spikes.sum()+ s['1'].sum()
    # pred = torch.argmax(summed_voltages, dim=1).to(device)
    pred = torch.argmax(summed_spikes, dim=1).to(device)
    # print(pred, target)
    correct += pred.eq(target.data.to(device))
    # correct += pred.eq(target).sum().item()
    accuracy = 100.0 * float(correct) / (index + 1)
    # print(correct)
    if index == 0:
        ciu = calculate_intermediate_usefulness(s['1'], SNN.connections["1","2"].w, target[0], time)
    else:
        ciu += calculate_intermediate_usefulness(s['1'], SNN.connections["1","2"].w, target[0], time)
    spikes_ = {
        layer: spikes[layer].get("s")[:].contiguous() for layer in spikes
    }
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
    #         ax.set_xlim(0,100)
    # voltage_ims, voltage_axes = plot_voltages(
    #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
    # )
    SNN.reset_state_variables()
print("ciu net: ",ciu)
# ciu = (ciu - ciu.mean())/ciu.std()
# ciu = torch.sigmoid(ciu)*1.2
# ciu = (ciu + min(ciu))
# ciu = ciu / max(ciu)
# ciu = ciu ** 2
import numpy as np
ciu = ((ciu - min(ciu))/(max(ciu) - min(ciu)))
ciu = 1-np.exp(-0.5 *ciu.cpu())
print(ciu)
# SNN_accuracy = 100.0 * float(correct) / len(train_loader2.dataset)
SNN_accuracy = 100.0 * float(correct) / num_data
print("Net spikes: ", net_spikes)


print("accuracy:, ", SNN_accuracy)

df = pd.DataFrame({"ANN accuracy":[ANN_accuracy],
                   "SNN accuracy": [SNN_accuracy]})
# ciu = ciu / (max(ciu)*0.9)
correct2=0
num_data2 = 0
net_spikes2 = 0
for i,num in enumerate(ciu):
    SNN.connections["0","1"].w[:,i] = SNN.connections["0","1"].w[:,i] * num * num
alg_pos = 0
alg_neg = 0
alg_mag = 0
for conn in set(SNN.connections.values()):
    alg_pos += torch.sum(conn.w[conn.w>0])
    alg_neg += torch.sum(conn.w[conn.w<0])
    alg_mag += torch.sum(torch.abs(conn.w))
for index, (data, target) in enumerate(train_loader2):
    # if index > 100:
    #     break
    num_data2 +=1
    print('sample ', index+1, 'elapsed', t_() - start)
    start = t_()

    data = data.to(device)
    data = data.view(-1, 3*32*32)
    inpts = {'Input': data.repeat(time, 1)}
    # print(inpts["Input"].shape)
    SNN.run(inputs=inpts, time=time)
    s = {layer: SNN.monitors[f'{layer}_spikes'].get('s') for layer in SNN.layers}
    # voltages = {layer: SNN.monitors[layer].get('v') for layer in ['2'] if not layer == 'Input'}
    # pred = torch.argmax(voltages['2'].sum(1))
    # summed_voltages = voltages['2'].sum(0)
    summed_spikes=s['2'].sum(0)
    net_spikes2 += summed_spikes.sum() + s['1'].sum()
    # pred = torch.argmax(summed_voltages, dim=1).to(device)
    pred = torch.argmax(summed_spikes, dim=1).to(device)
    # print(pred, target)
    # correct += pred.eq(target.data.to(device)).cpu().sum()
    correct2 += pred.eq(target).sum().item()
    accuracy = 100.0 * float(correct) / (index + 1)
    # print(correct)
    # if index == 0:
    #     ciu = calculate_intermediate_usefulness(s['1'], SNN.connections["1","2"].w, target[0], time)
    # else:
    #     ciu += calculate_intermediate_usefulness(s['1'], SNN.connections["1","2"].w, target[0], time)
    spikes_ = {
        layer: spikes[layer].get("s")[:].contiguous() for layer in spikes
    }
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
    SNN.reset_state_variables()
SNN_accuracy = 100.0 * float(correct) / num_data
SNN_accuracy2 = 100.0 * float(correct2) / num_data2

print("accuracy 1:, ", SNN_accuracy, "accuracy reduced spikes: ", SNN_accuracy2)
print("net_spikes: ", net_spikes, "net_spikes reduced spikes: ", net_spikes2)
print("net_spikes adjusted for #images: ", net_spikes/num_data, "net_spikes reduced spikes #im adjusted: ", net_spikes2/num_data2)
print(f"baseline weights pos: {baseline_pos} neg: {baseline_neg} mag {baseline_mag}")
print(f"algo weights pos: {alg_pos} neg: {alg_neg} mag {alg_mag}")
df.to_csv("accuracy_hidden_1.csv")