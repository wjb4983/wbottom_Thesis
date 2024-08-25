import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from torchvision import datasets, transforms

from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.network import Network
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network.nodes import Input

import bindsnet.network.nodes as nodes
import bindsnet.network.topology as topology
from bindsnet.conversion.nodes import PassThroughNodes, SubtractiveResetIFNodes
from bindsnet.conversion.topology import ConstantPad2dConnection, PermuteConnection
from bindsnet.network import Network
# from bindsnet.conversion import ann_to_snn_helper, data_based_normalization, _ann_to_snn_helper
from bindsnet.conversion.conversion import data_based_normalization, _ann_to_snn_helper

from copy import deepcopy
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

#########################################################
#I changed the name of the input layer to be "0" no matter
#I can't remember why, but I think there was weird behavior
def ann_to_snn(
    ann: Union[nn.Module, str],
    input_shape: Sequence[int],
    data: Optional[torch.Tensor] = None,
    percentile: float = 99.9,
    node_type: Optional[nodes.Nodes] = SubtractiveResetIFNodes,
    **kwargs,
) -> Network:
    # language=rst
    """
    Converts an artificial neural network (ANN) written as a
    ``torch.nn.Module`` into a near-equivalent spiking neural network.

    :param ann: Artificial neural network implemented in PyTorch. Accepts
        either ``torch.nn.Module`` or path to network saved using
        ``torch.save()``.
    :param input_shape: Shape of input data.
    :param data: Data to use to perform data-based weight normalization of
        shape ``[n_examples, ...]``.
    :param percentile: Percentile (in ``[0, 100]``) of activations to scale by
        in data-based normalization scheme.
    :param node_type: Class of ``Nodes`` to use in replacing
        ``torch.nn.Linear`` layers in original ANN.
    :return: Spiking neural network implemented in PyTorch.
    """
    if isinstance(ann, str):
        ann = torch.load(ann)
    else:
        ann = deepcopy(ann)

    assert isinstance(ann, nn.Module)

    if data is None:
        import warnings

        warnings.warn("Data is None. Weights will not be scaled.", RuntimeWarning)
    else:
        ann = data_based_normalization(
            ann=ann, data=data.detach(), percentile=percentile
        )

    snn = Network()

    input_layer = nodes.Input(shape=input_shape, traces = True, tc_trace=20.0)
    snn.add_layer(input_layer, name="0")

    children = []
    for c in ann.children():
        if isinstance(c, nn.Sequential):
            for c2 in list(c.children()):
                children.append(c2)
        else:
            children.append(c)

    i = 0
    prev = input_layer
    while i < len(children) - 1:
        current, nxt = children[i : i + 2]
        layer, connection = _ann_to_snn_helper(prev, current, node_type, **kwargs)

        i += 1

        if layer is None or connection is None:
            continue

        snn.add_layer(layer, name=str(i))
        snn.add_connection(connection, source=str(i - 1), target=str(i))

        prev = layer

    current = children[-1]
    layer, connection = _ann_to_snn_helper(
        prev, current, node_type, last=True, **kwargs
    )

    i += 1

    if layer is not None or connection is not None:
        snn.add_layer(layer, name=str(i))
        snn.add_connection(connection, source=str(i - 1), target=str(i))

    return snn

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=100)
parser.add_argument("--n_train", type=int, default=0)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=8000)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=.05)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=0.001)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--num_layers", type=int, default=2)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
n_updates = args.n_updates
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu
num_layers = args.num_layers

# update_steps = int(n_train / batch_size / n_updates)
# update_interval = update_steps * batch_size
# update_steps = 20
# update_interval = 20
# print(update_interval)

device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity


def binarize_image(data):
    data = (data > 0.5).float()
    return data
# from bindsnet.utils import load_network

# network.load_state_dict(torch.load('snn_conversion.pth'))
from VGGSmall_SNNConversion import VGGSmall
network = VGGSmall(num_classes=10)
# network.load_state_dict(torch.load('presnn_conversion.pth'))trained_model.pt
network.load_state_dict(torch.load('trained_model.pt'))
# print(network)
import torch.nn as nn
assert isinstance(network, nn.Module)
from bindsnet.network.nodes import AdaptiveLIFNodes
# from bindsnet.conversion import ann_to_snn
network = ann_to_snn(network,(1,28,28))#, node_type=AdaptiveLIFNodes, theta_plus=0.05)# (3,32,32)
xp = network.children()
for c in xp:
    print(c)
def zero_out_conv2d_biases(snn):
    for connection in snn.connections.values():
        if isinstance(connection, nn.Conv2d):
            with torch.no_grad():
                if connection.bias is not None:
                    connection.bias.fill_(0)
zero_out_conv2d_biases(network)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
# dataset = MNIST(
#     PoissonEncoder(time=time, dt=dt),
#     None,
#     "../../../data/MNIST",
#     download=True,
#     transform=transforms.Compose(
#         [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
#     ),
# )

class FlattenTransform:
    def __call__(self, x):
        return x.view(-1, 28*28)


train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               # transform=transforms.Compose([transforms.ToTensor(),FlattenTransform()]))
                               transform=transforms.ToTensor())

train_dataset2 = datasets.MNIST('./data',
                               train=False,
                               download=True,
                               # transform=transforms.Compose([transforms.ToTensor(),FlattenTransform()]))
                               transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True, batch_size=train_dataset.__len__(), generator=torch.Generator(device=device))

train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,
                                           shuffle=True,)
                                           # generator=torch.Generator(device=device))

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)
per_class = int(n_neurons/n_classes)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}



inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((batch_size, int(time / dt), n_neurons), device=device)

# Train the network.
i=0
last_layer = None
layers = {}
# print(network.layers)
for layer_name in network.layers:

    # print(layer_name)
    last_layer=layer_name
    # if(layer_name == "0"):
    #     network.layers["Input"] = Input(
    #             n=784, shape=(784,), traces=False,
    #         )
for connection in network.connections.values():
    if "MaxPool2d" not in connection.__class__.__name__:
        print(f"Connection Name: {connection.source, connection.target}")
        print(f"Weight Matrix Shape: {connection.w.shape}")
        if(i==0):
            print(f"Weight matrix: {connection.w}")
        print(f"{connection.type}")
        print("==============================")
    else:
        print("=================\nMaxpool\n==================\n")
        print(f"{connection.type}")
    connection.reduction = torch.sum
    i=i+1
        

ims = [None] * len(set(network.layers))
axes = [None] * len(set(network.layers))
spike_ims, spike_axes = None, None
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

        
network.add_monitor(
    Monitor(network.layers[f'{last_layer}'], state_vars=['v'], time=time), name=f'{last_layer}'
)

print("\nBegin training...")
start = t()


# Load MNIST data.
# test_dataset = MNIST(
#     PoissonEncoder(time=time, dt=dt),
#     None,
#     root=os.path.join(ROOT_DIR, "data", "MNIST"),
#     download=True,
#     train=False,
#     transform=transforms.Compose(
#         [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
#     ),
# )

# # Create a dataloader to iterate and batch data
# test_dataloader = DataLoader(
#     test_dataset,
#     batch_size=batch_size,
#     shuffle=False,
#     num_workers=n_workers,
#     pin_memory=gpu,
# )

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing...\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
pbar.set_description_str("Test progress: ")
correct=0
for connection in network.connections.values():
    if hasattr(connection, 'w'):
        print(f"Mean weight of connection {connection}: {torch.mean(connection.w)}")
        print(f"min: {connection.w.min()} max: {connection.w.max()}")
for step, batch in enumerate(train_loader2):
    if step * batch_size > n_test:
        break
    data = batch[0]
    target=batch[1]
    # Get next input sample.
    # inputs = {"Input": batch["encoded_image"]}
    data = data.to(device)
    data = (data - data.min()) / (data.max() - data.min())
    # Binarize the data
    data = binarize_image(data)
    # data = data.view(-1, 28*28)
    inputs = {'0': data.repeat(time,1,1, 1,1)}
    print(inputs['0'].shape)
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    # for key, tensor in inputs.items():
    #     non_zero_indices = (tensor != 0).nonzero(as_tuple=True)
    #     num_non_zero_elements = len(non_zero_indices[0])
    #     print(f"Number of non-zero elements for {key}: {num_non_zero_elements} / {tensor.shape}")
    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record = spikes[f"{last_layer}"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(target, device=device)

    # Get network predictions.
    # all_activity_pred = all_activity(
    #     spikes=spike_record, assignments=assignments, n_labels=n_classes
    # )
    # proportion_pred = proportion_weighting(
    #     spikes=spike_record,
    #     assignments=assignments,
    #     proportions=proportions,
    #     n_labels=n_classes,
    # )
    # print(label_tensor, all_activity_pred)
    # Compute network accuracy according to available classification strategies.
    voltages = {layer: network.monitors[layer].get('v') for layer in [f'{last_layer}'] if not layer == 'Input'}
    summed_voltages = voltages[f'{last_layer}'].sum(0)
    pred = torch.argmax(summed_voltages, dim=1).to(device)
    print("\n",pred, target)
    # correct += pred.eq(target.data.to(device)).cpu().sum()
    correct += pred.eq(label_tensor).sum().item()
    accuracy["all"] += float(
        torch.sum(label_tensor.long() == correct).item()
    )
    
    spikes_ = {
        layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
    }
    # spikes_.update({"Input": inputs["X"]})
    # print(spikes)
    # p,o = plot_spikes(
    #     {"i": inputs["X"]},
    #     ims=ims[-1], axes=axes[-1]
    # )
    keys = list(spikes_.keys())
    for i in range(0, len(keys), 2):
        # Get two consecutive layers from spikes_
        
        layer1_key = keys[i]
        layer2_key = keys[i + 1] if i + 1 < len(keys) else None
        
        # Get the spike data for the current layers
        layer1_spikes = spikes_[layer1_key]
        layer2_spikes = spikes_[layer2_key] if layer2_key else None
        if(layer2_spikes == None):
            #ims[i], axes[i]
            x, y = plot_spikes(
                {layer1_key: layer1_spikes},
                ims=ims[i], axes=axes[i]
            )
        else:
            # ims[i], axes[i]
            x, y= plot_spikes(
                {layer1_key: layer1_spikes, layer2_key: layer2_spikes},
                ims=ims[i], axes=axes[i]
            )

    network.reset_state_variables()  # Reset state variables.
    pbar.update(batch_size)
pbar.close()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

# print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTesting complete.\n")