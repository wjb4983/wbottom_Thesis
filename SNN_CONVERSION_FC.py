import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

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
from bindsnet.encoding import PoissonEncoder, RepeatEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes

from copy import deepcopy
from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

import bindsnet.network.nodes as nodes
import bindsnet.network.topology as topology
from bindsnet.conversion.nodes import PassThroughNodes, SubtractiveResetIFNodes
from bindsnet.conversion.topology import ConstantPad2dConnection, PermuteConnection
from bindsnet.network import Network
# from bindsnet.conversion import ann_to_snn_helper, data_based_normalization, _ann_to_snn_helper
from bindsnet.conversion.conversion import data_based_normalization, _ann_to_snn_helper
class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 100, bias=False)
        self.fc2 = nn.Linear(100, 10, bias=False)
        # self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        return x



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

#####################################################################
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=500)
parser.add_argument("--n_train", type=int, default=0)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=6000)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=.05)
parser.add_argument("--time", type=int, default=150)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=1)#0.000001)
parser.add_argument("--progress_interval", type=int, default=100)
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
##################################################################################
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



# from bindsnet.utils import load_network

# network.load_state_dict(torch.load('snn_conversion.pth'))
from ANNFC import FCNetworkHC
from bindsnet.learning.learning import PostPre
from bindsnet.network.topology import Connection
# network = FCNetworkHC(1*28*28, 100, 10, 0.0, 1)
network = FCModel()

network.load_state_dict(torch.load("trained_model.pt"))
# network.load_state_dict(torch.load('presnn_conversionFC.pth'))
import torch.nn as nn
assert isinstance(network, nn.Module)
# from bindsnet.conversion import ann_to_snn
# Load MNIST data.
test_dataset = MNIST(
    # PoissonEncoder(time=time, dt=dt),
    RepeatEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..","..", "data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)
# for d, target in test_dataset:
#     data = d.to(device)
network = ann_to_snn(network, (1, 28,28), data=None,node_type = AdaptiveLIFNodes, theta_plus=0.05)

from bindsnet.network.nodes import Input


# Directs network to GPU
if gpu:
    network.to("cuda")


# Neuron assignments and spike proportions.
n_classes = 10

#TODO Changed n_neurons to 10 - need to make variable called num_outputs, or use network itself
assignments = -torch.ones(10, device=device)
proportions = torch.zeros((10, n_classes), device=device)
rates = torch.zeros((10, n_classes), device=device)
per_class = int(10/n_classes)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}



inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((batch_size * 25, int(time / dt), 10), device=device)

# Train the network.
i=0
last_layer = None
for layer_name in network.layers:

    print(layer_name)
    last_layer=layer_name
    #This changes attributes about the network layers - makes it like Diehl and Cook example but not necessary
    # if(layer_name != "0"):
        # print(network.layers[layer_name].v)
        # # network.layers[layer_name].v = torch.tensor(-60.0)
        # network.layers[layer_name].register_buffer("thresh", torch.tensor(-54.0))
        # network.layers[layer_name].register_buffer("reset", torch.tensor(-65.0))
        # network.layers[layer_name].register_buffer("theta_plus", torch.tensor(100))

for connection in network.connections.values():
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
    if batch_size > 1:
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


one_V = Monitor(
    network.layers["1"], ["v"], time=int(time / dt), device=device
)
two_V = Monitor(
    network.layers["2"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(one_V, name="1")
network.add_monitor(two_V, name="2")
voltages = {}
for layer in set(network.layers) - {"0"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)
        
        
print("\nBegin training...")
start = t()




# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=gpu,
)

#If we want to ensure weights are > 0 - Diehl and Cook example only uses > 0 weights
#for non inhibitory
with torch.no_grad():
    network.connections[("0", "1")].wmin=torch.nn.Parameter(torch.tensor(0.0))
    network.connections[("1", "2")].w.copy_(torch.clamp(network.connections[("1", "2")].w, min=0))


# Train the network.
print("\nBegin testing...\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
pbar.set_description_str("Test progress: ")
labels=[]
for step, batch in enumerate(test_dataloader):
    if step * batch_size > n_test:
        break
    # Get next input sample.
    # data = data.to(device)
    # data = data.view(-1, 28*28)
    # inputs = {'Input': data.repeat(time, 1)}
    inputs = {"0": batch["encoded_image"]}
    # print(inputs["X"].shape)
    # inputs["X"] = torch.ones_like(inputs["X"])
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}
        
    if step % 25 == 0 and step > 0:
        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(labels, device=device)

        # Get network predictions.
        all_activity_pred = all_activity(
            spikes=spike_record, assignments=assignments, n_labels=n_classes
        )
        proportion_pred = proportion_weighting(
            spikes=spike_record,
            assignments=assignments,
            proportions=proportions,
            n_labels=n_classes,
        )

        # Compute network accuracy according to available classification strategies.
        accuracy["all"].append(
            100
            * torch.sum(label_tensor.long() == all_activity_pred).item()
            / len(label_tensor)
        )
        accuracy["proportion"].append(
            100
            * torch.sum(label_tensor.long() == proportion_pred).item()
            / len(label_tensor)
        )

        print(
            "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (
                accuracy["all"][-1],
                np.mean(accuracy["all"]),
                np.max(accuracy["all"]),
            )
        )
        print(
            "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
            " (best)\n"
            % (
                accuracy["proportion"][-1],
                np.mean(accuracy["proportion"]),
                np.max(accuracy["proportion"]),
            )
        )

        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(
            spikes=spike_record,
            labels=label_tensor,
            n_labels=n_classes,
            rates=rates,
        )

        labels = []
    # for key, tensor in inputs.items():
    #     non_zero_indices = (tensor != 0).nonzero(as_tuple=True)
    #     num_non_zero_elements = len(non_zero_indices[0])
    #     print(f"Number of non-zero elements for {key}: {num_non_zero_elements} / {tensor.shape}")
    # Run the network on the input.
    # for key, tensor in inputs.items():
    #     non_zero_indices = (tensor != 0).nonzero(as_tuple=True)
    #     num_non_zero_elements = len(non_zero_indices[0])
    #     print(f"Number of non-zero elements for {key}: {num_non_zero_elements} / {tensor.shape[0]*tensor.shape[1]*tensor.shape[2]*tensor.shape[3]*tensor.shape[4]}")
    # network2.run(inputs=inputs, time=time)
    network.run(inputs=inputs, time=time)


    # Add to spikes recording.
    s = spikes[f"{last_layer}"].get("s").permute((1, 0, 2))
    spike_record[
        (step * batch_size)
        % 25 : (step * batch_size % 25)
        + s.size(0)
    ] = s
    # spike_record = spikes2[f"out"].get("s").permute((1, 0, 2))
    
    exc_voltages = one_V.get("v")
    inh_voltages = two_V.get("v")
    voltages = {"1": exc_voltages, "2": inh_voltages}

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)
    labels.extend(batch["label"].tolist())
    # label_tensor = torch.tensor(label, device=device)

    # Get network predictions.
    # all_activity_pred = all_activity(
    #     spikes=s, assignments=assignments, n_labels=n_classes
    # )
    # proportion_pred = proportion_weighting(
    #     spikes=s,
    #     assignments=assignments,
    #     proportions=proportions,
    #     n_labels=n_classes,
    # )

    # # Compute network accuracy according to available classification strategies.
    # accuracy["all"] += float(
    #     torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
    # )
    # accuracy["proportion"] += float(
    #     torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
    # )
    
    # spikes_ = {
    #     layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
    # }
    spikes_ = {
        layer: spikes[layer].get("s")[:].contiguous() for layer in spikes
    }
    # spikes_ = {
    #     layer: spikes2[layer].get("s")[:, 0].contiguous() for layer in spikes2
    # }
    keys = list(spikes_.keys())
    for i in range(0, len(keys), 2):
        # Get two consecutive layers from spikes_
        
        layer1_key = keys[i]
        layer2_key = keys[i + 1] if i + 1 < len(keys) else None
        
        # Get the spike data for the current layers
        layer1_spikes = spikes_[layer1_key]
        layer2_spikes = spikes_[layer2_key] if layer2_key else None
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
    voltage_ims, voltage_axes = plot_voltages(
        voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
    )
    

    # w1 = network.connections[("0", "1")].w
    # print(w1.mean().item())

    network.reset_state_variables()  # Reset state variables.
    pbar.update(batch_size)
pbar.close()

# print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
# print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))
print(
    "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
    % (
        accuracy["all"][-1],
        np.mean(accuracy["all"]),
        np.max(accuracy["all"]),
    )
)
print(
    "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
    " (best)\n"
    % (
        accuracy["proportion"][-1],
        np.mean(accuracy["proportion"]),
        np.max(accuracy["proportion"]),
    )
)

# print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTesting complete.\n")