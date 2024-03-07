# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 12:27:14 2024

@author: wbott
"""

import argparse
import os
from time import time as t

import numpy as np
import torch
from torchvision import transforms
from tqdm.auto import tqdm
# from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import CIFAR10
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network import Network

from MultilayerDiehlandCook import MultiLayerDiehlAndCook2015

# class MultiLayerDiehlAndCook2015(Network):
#     def __init__(
#         self,
#         n_inpt: int,
#         n_neurons: int = 100,
#         num_layers: int = 10,
#         exc: float = 22.5,
#         inh: float = 17.5,
#         dt: float = 1.0,
#         nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
#         reduction: Optional[callable] = None,
#         wmin: float = 0.0,
#         wmax: float = 1.0,
#         norm: float = 78.4,
#         theta_plus: float = 0.05,
#         tc_theta_decay: float = 1e7,
#         inpt_shape: Optional[Iterable[int]] = None,
#         inh_thresh: float = -40.0,
#         exc_thresh: float = -52.0,
#     ) -> None:
#         super().__init__(dt=dt)

#         self.n_inpt = n_inpt
#         self.inpt_shape = inpt_shape
#         self.n_neurons = n_neurons
#         self.num_layers = num_layers
#         self.exc = exc
#         self.inh = inh
#         self.dt = dt

#         # Layers and connections lists
#         self.input_layers = []
#         self.exc_layers = []
#         self.inh_layers = []
#         self.input_exc_connections = []
#         self.exc_inh_connections = []
#         self.inh_exc_connections = []

#         for i in range(self.num_layers):
#             # Create input layer
#             input_layer = Input(
#                 n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
#             )
#             self.input_layers.append(input_layer)

#             # Create excitatory layer
#             exc_layer = DiehlAndCookNodes(
#                 n=self.n_neurons,
#                 traces=True,
#                 rest=-65.0,
#                 reset=-60.0,
#                 thresh=exc_thresh,
#                 refrac=5,
#                 tc_decay=100.0,
#                 tc_trace=20.0,
#                 theta_plus=theta_plus,
#                 tc_theta_decay=tc_theta_decay,
#             )
#             self.exc_layers.append(exc_layer)

#             # Create inhibitory layer
#             inh_layer = LIFNodes(
#                 n=self.n_neurons,
#                 traces=False,
#                 rest=-60.0,
#                 reset=-45.0,
#                 thresh=inh_thresh,
#                 tc_decay=10.0,
#                 refrac=2,
#                 tc_trace=20.0,
#             )
#             self.inh_layers.append(inh_layer)

#             # Create connections
#             w_input_exc = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
#             input_exc_conn = Connection(
#                 source=input_layer,
#                 target=exc_layer,
#                 w=w_input_exc,
#                 update_rule=PostPre,
#                 nu=nu,
#                 reduction=reduction,
#                 wmin=wmin,
#                 wmax=wmax,
#                 norm=norm,
#             )
#             self.input_exc_connections.append(input_exc_conn)

#             w_exc_inh = self.exc * torch.diag(torch.ones(self.n_neurons))
#             exc_inh_conn = Connection(
#                 source=exc_layer, target=inh_layer, w=w_exc_inh, wmin=0, wmax=self.exc
#             )
#             self.exc_inh_connections.append(exc_inh_conn)

#             w_inh_exc = -self.inh * (
#                 torch.ones(self.n_neurons, self.n_neurons)
#                 - torch.diag(torch.ones(self.n_neurons))
#             )
#             inh_exc_conn = Connection(
#                 source=inh_layer, target=exc_layer, w=w_inh_exc, wmin=-self.inh, wmax=0
#             )
#             self.inh_exc_connections.append(inh_exc_conn)

#             # Add layers and connections to the network
#             self.add_layer(input_layer, name=f"X_{i}")
#             self.add_layer(exc_layer, name=f"Ae_{i}")
#             self.add_layer(inh_layer, name=f"Ai_{i}")
#             self.add_connection(input_exc_conn, source=f"X_{i}", target=f"Ae_{i}")
#             self.add_connection(exc_inh_conn, source=f"Ae_{i}", target=f"Ai_{i}")
#             self.add_connection(inh_exc_conn, source=f"Ai_{i}", target=f"Ae_{i}")


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=100)
parser.add_argument("--n_train", type=int, default=1000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=100)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
# 1 if you want new model, 0 if use pretrained model
parser.add_argument("--new_model", type=int, default=0)
parser.add_argument("--num_layers", type=int, default=2)
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
new_model = args.new_model
num_layers = args.num_layers

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
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = MultiLayerDiehlAndCook2015(
    n_inpt=32 * 32 * 3,
    n_neurons=n_neurons,
    num_layers=num_layers,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=307.2,  # 78.4,
    theta_plus=theta_plus,
    inpt_shape=(3, 32, 32),
)

if os.path.isfile("MultidiehlcookCIFAR-10.pth")  and new_model == 1:
    print("=======================================\nUsing MultidiehlcookCIFAR-10.pth found on your computer\n============================")
    # network.load_state_dict(torch.load('MultidiehlcookCIFAR-10.pth'))
else:
    print("=======================================\nCreating new model - saved as MultidiehlcookCIFAR-10.pth\n============================")
    # note model created above
# Directs network to GPU
if gpu:
    network.to("cuda")

train_dataset = CIFAR10(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "..", "data", "CIFAR10"),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Record spikes during the simulation.
spike_record = torch.zeros(
    (update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
# exc_voltage_monitor = Monitor(
#     network.layers["Ae"], ["v"], time=int(time / dt), device=device
# )
# inh_voltage_monitor = Monitor(
#     network.layers["Ai"], ["v"], time=int(time / dt), device=device
# )
# network.add_monitor(exc_voltage_monitor, name="exc_voltage")
# network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

# voltages = {}
# for layer in set(network.layers) - {"X"}:
#     voltages[layer] = Monitor(
#         network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
#     )
#     network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Train the network.
print("\nBegin training.\n")
start = t()
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" %
              (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(
            int(time / dt), 1, 3, 32, 32)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
            non_negative_indices = torch.nonzero(assignments != -1).squeeze()
            # Print the indices and corresponding values
            for index in non_negative_indices:
                print(f"Index: {index}, Assignment: {assignments[index]}")
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
            print(label_tensor)
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
            non_negative_indices = torch.nonzero(assignments != -1).squeeze()
            # Print the indices and corresponding values
            for index in non_negative_indices:
                print(f"Index: {index}, Assignment: {assignments[index]}")

        labels.append(batch["label"])

        # Run the network on the input.
        network.run(inputs=inputs, time=time)

        # Get voltage recording.
        # exc_voltages = exc_voltage_monitor.get("v")
        # inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        spike_record[step %
                     update_interval] = spikes[f"Ae_{num_layers-1}"].get("s").squeeze()

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(3, 32, 32).permute(1, 2, 0)
            image = image / image.max()
            inpt = inputs["X"].view(time, 3072).sum(0).view(32, 32, 3)
            # input_exc_weights = network.connections[("X", "Ae_0")].w
            # square_weights = get_square_weights(
            #     input_exc_weights.view(3072, n_neurons), n_sqrt, 32
            # )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            # voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(
                spikes_, ims=spike_ims, axes=spike_axes)
            # weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(
                accuracy, x_scale=update_interval, ax=perf_ax)
            # voltage_ims, voltage_axes = plot_voltages(
            #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            # )

            # plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        # Find indices where assignments doesn't equal -1

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Load CIFAR-10 test data
test_dataset = CIFAR10(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "..", "data", "CIFAR10"),
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test, position=0, leave=True)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(
        int(time / dt), 1, 3, 32, 32).to(device)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record[0] = spikes[f"Ae_{num_layers-1}"].get("s").squeeze()

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
    accuracy["all"] += float(torch.sum(label_tensor.long()
                             == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" %
      (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

# Save the trained network weights
torch.save(network.state_dict(), 'MultidiehlcookCIFAR-10.pth')
