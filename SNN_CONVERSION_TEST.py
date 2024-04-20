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
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_neurons", type=int, default=64)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=0)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=8000)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=.05)
parser.add_argument("--time", type=int, default=150)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
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



# from bindsnet.utils import load_network

# network.load_state_dict(torch.load('snn_conversion.pth'))
from VGGSmall_SNNConversion import VGGSmall
network = VGGSmall(num_classes=10)
network.load_state_dict(torch.load('presnn_conversion.pth'))
# print(network)
import torch.nn as nn
assert isinstance(network, nn.Module)
from bindsnet.conversion import ann_to_snn
network = ann_to_snn(network, (1,28,28))
xp = network.children()
for c in xp:
    print(c)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../../data/MNIST",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

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
    print(layer_name)
    last_layer=layer_name
for connection in network.connections.values():
    if "MaxPool2d" not in connection.__class__.__name__:
        print(f"Connection Name: {connection.source, connection.target}")
        print(f"Weight Matrix Shape: {connection.w.shape}")
        print(f"{connection.type}")
        print("==============================")
    else:
        print("=================\nMaxpool\n==================\n")
        print(f"{connection.type}")
    connection.reduction = torch.sum
        

ims = [None] * len(set(network.layers))
axes = [None] * len(set(network.layers))
spike_ims, spike_axes = None, None
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

        
        
print("\nBegin training...")
start = t()


# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join(ROOT_DIR, "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Create a dataloader to iterate and batch data
test_dataloader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=n_workers,
    pin_memory=gpu,
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing...\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
pbar.set_description_str("Test progress: ")

for step, batch in enumerate(test_dataloader):
    if step * batch_size > n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"]}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}
    for key, tensor in inputs.items():
        non_zero_indices = (tensor != 0).nonzero(as_tuple=True)
        num_non_zero_elements = len(non_zero_indices[0])
        print(f"Number of non-zero elements for {key}: {num_non_zero_elements}")
    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record = spikes[f"{last_layer}"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

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
    accuracy["all"] += float(
        torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
    )
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
    )
    
    spikes_ = {
        layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
    }
    keys = list(spikes_.keys())
    for i in range(0, len(keys), 3):
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