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
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--n_updates", type=int, default=60)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
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

update_steps = int(n_train / batch_size / n_updates)
update_interval = update_steps * batch_size
print(update_interval)

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

# Build network.
# network = DiehlAndCook2015(
#     n_inpt=784,
#     n_neurons=n_neurons,
#     exc=exc,
#     inh=inh,
#     dt=dt,
#     norm=78.4,
#     nu=(1e-4, 1e-2),
#     theta_plus=theta_plus,
#     inpt_shape=(1, 28, 28),
# )

from MultilayerDiehlandCook import MultiLayerDiehlAndCook2015
network = MultiLayerDiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    num_layers=num_layers,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

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

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers[f"Ae_{num_layers-2}"], ["v"], time=int(time / dt), device=device
)
# inh_voltage_monitor = Monitor(
#     network.layers[f"Ai_{num_layers-1}"], ["v"], time=int(time / dt), device=device
# )
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
# network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
# spikes = {}
# for layer in set(network.layers):
#     spikes[layer] = Monitor(
#         network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
#     )
#     network.add_monitor(spikes[layer], name="%s_spikes" % layer)

ims = [None] * len(set(network.layers))
axes = [None] * len(set(network.layers))
spike_ims, spike_axes = None, None
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)


voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.

for connection_name, connection in network.connections.items():
    print(f"Connection Name: {connection_name}")
    
    source_name = getattr(connection.source, "name", None)
    target_name = getattr(connection.target, "name", None)
    
    print(f"Source Layer: {source_name}")
    print(f"Target Layer: {target_name}")
    print(f"Weight Matrix Shape: {connection.w.shape}")
    print("=" * 30)
    
    
print("\nBegin training...")
start = t()

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("\nProgress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    pbar_training = tqdm(total=n_train)
    for step, batch in enumerate(train_dataloader):
        if step * batch_size > n_train:
            break

        # Assign labels to excitatory neurons.
        if step % update_steps == 0 and step > 0:
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
                * torch.sum(label_tensor.long() == all_activity_pred.to(device)).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred.to(device)).item()
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

        # Get next input sample.
        inputs = {"X": batch["encoded_image"]}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Remember labels.
        labels.extend(batch["label"].tolist())

        # Run the network on the input.
        network.run(inputs=inputs, time=time)

        # Add to spikes recording.
        s = spikes[f"Ae_{num_layers-2}"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        # inh_voltages = inh_voltage_monitor.get("v")

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"][:, 0].view(28, 28)
            inpt = inputs["X"][:, 0].view(time, 784).sum(0).view(28, 28)
            lable = batch["label"][0]
            input_exc_weights = network.connections[("X", f"Ae_0")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {
                layer: spikes[layer].get("s")[:, 0].contiguous() for layer in spikes
            }
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
#################################################################################
            voltages = {f"Ae_{num_layers-2}": exc_voltages}#, f"Ai_{num_layers-1}": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=lable, axes=inpt_axes, ims=inpt_ims
            )
            # spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(
                accuracy, x_scale=update_steps * batch_size, ax=perf_ax
            )
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            # plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)
    pbar_training.close()

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTraining complete.\n")

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
    shuffle=True,
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

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record = spikes[f"Ae_{num_layers-1}"].get("s").permute((1, 0, 2))

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

    network.reset_state_variables()  # Reset state variables.
    pbar.update(batch_size)
pbar.close()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("\nTesting complete.\n")