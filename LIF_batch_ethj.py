import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
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
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network import Network

from typing import Iterable, Optional, Sequence, Union


from bindsnet.learning import PostPre
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes, AdaptiveLIFNodes
from bindsnet.network.topology import Connection

from MultilayerDiehlandCook import MultiLayerDiehlAndCook2015
from plot_spikes_custom import plot_spikes_custom

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120.0)
parser.add_argument("--theta_plus", type=float, default=0.7)
parser.add_argument("--time", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=400)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=2000)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--new_model", type=int, default=0) #1 if you want new model, 0 if use pretrained model
parser.add_argument("--batch_size", type=int, default=250)
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
train = True
plot = args.plot
gpu = args.gpu
new_model = args.new_model
batch_size = args.batch_size
plot=True

fig = []

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

class LIFNET(Network):
    def __init__(        self,
            n_inpt: int,
            n_neurons: int = 100,
            exc: float = 22.5,
            inh: float = 17.5,
            dt: float = 1.0,
            nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
            reduction: Optional[callable] = None,
            wmin: float = 0.0,
            wmax: float = 1.0,
            norm: float = 78.4,
            theta_plus: float = 0.05,
            tc_theta_decay: float = 1e7,
            inpt_shape: Optional[Iterable[int]] = None,
            inh_thresh: float = -40.0,
            exc_thresh: float = -52.0,):
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt

        # Layers
        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        exc_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        inh_layer = LIFNodes(
            n=self.n_neurons,
            traces=False,
            rest=-60.0,
            reset=-45.0,
            thresh=inh_thresh,
            tc_decay=10.0,
            refrac=2,
            tc_trace=20.0,
        )

        # Connections
        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_exc_conn = Connection(
            source=input_layer,
            target=exc_layer,
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=10.0,
            norm=norm,
        )
        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        exc_inh_conn = Connection(
            source=exc_layer, target=inh_layer, w=w, wmin=0, wmax=self.exc
        )
        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        inh_exc_conn = Connection(
            source=inh_layer, target=exc_layer, w=w, wmin=-self.inh, wmax=0
        )

        # Add to network
        self.add_layer(input_layer, name="X")
        self.add_layer(exc_layer, name="Ae")
        self.add_layer(inh_layer, name="Ai")
        self.add_connection(input_exc_conn, source="X", target="Ae")
        self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        self.add_connection(inh_exc_conn, source="Ai", target="Ae")
        # super().__init__(dt=dt)
        # self.n_inpt = n_inpt
        # self.inpt_shape = inpt_shape
        # self.n_neurons = n_neurons

# Initialize the network
# network = LIFNET(
#     n_inpt=784, n_neurons=args.n_neurons, exc=args.exc, inh=args.inh, dt=args.dt,
#     theta_plus=args.theta_plus, tc_theta_decay=1e7, inpt_shape=(1, 28, 28)
        # # Layers
        # input_layer = Input(n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0)
        # exc_layer = DiehlAndCookNodes(
        #     n=self.n_neurons, traces=True, rest=-65.0, reset=-60.0, thresh=-52.0,
        #     refrac=5, tc_decay=10.0, tc_trace=20.0, theta_plus=theta_plus#, tc_theta_decay=tc_theta_decay
        # )
        # inh_layer = LIFNodes(
        #     n=self.n_neurons, traces=False, rest=-60.0, reset=-45.0, thresh=-40.0,
        #     tc_decay=10.0, refrac=2
        # )

        # # Connections
        # input_exc_conn = Connection(
        #     source=input_layer, target=exc_layer, w=1.0 * torch.rand(self.n_inpt, self.n_neurons),
        #     update_rule=PostPre, nu=(1e-4, 1e-2), wmin=0.0, wmax=1.0, norm=78.4
        # )
        # exc_inh_conn = Connection(source=exc_layer, target=inh_layer, w=exc * torch.diag(torch.ones(self.n_neurons)))
        # inh_exc_conn = Connection(source=inh_layer, target=exc_layer, w=-inh * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons))))

        # # Add layers and connections to the network
        # self.add_layer(input_layer, name="X")
        # self.add_layer(exc_layer, name="Ae")
        # self.add_layer(inh_layer, name="Ai")
        # self.add_connection(input_exc_conn, source="X", target="Ae")
        # self.add_connection(exc_inh_conn, source="Ae", target="Ai")
        # self.add_connection(inh_exc_conn, source="Ai", target="Ae")
class DiehlAndCook2015v2(Network):
    # language=rst
    """
    Slightly modifies the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_ by removing
    the inhibitory layer and replacing it with a recurrent inhibitory connection in the
    output layer (what used to be the excitatory layer).
    """

    def __init__(
        self,
        n_inpt: int,
        n_neurons: int = 100,
        inh: float = 17.5,
        dt: float = 1.0,
        nu: Optional[Union[float, Sequence[float]]] = (1e-4, 1e-2),
        reduction: Optional[callable] = None,
        wmin: Optional[float] = 0.0,
        wmax: Optional[float] = 1.0,
        norm: float = 78.4,
        theta_plus: float = 0.05,
        tc_theta_decay: float = 1e7,
        inpt_shape: Optional[Iterable[int]] = None,
        exc_thresh: float = -52.0,
    ) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015v2``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization
            constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane
            threshold potential.
        :param tc_theta_decay: Time constant of ``DiehlAndCookNodes`` threshold
            potential decay.
        :param inpt_shape: The dimensionality of the input layer.
        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.inpt_shape = inpt_shape
        self.n_neurons = n_neurons
        self.inh = inh
        self.dt = dt

        input_layer = Input(
            n=self.n_inpt, shape=self.inpt_shape, traces=True, tc_trace=20.0
        )
        self.add_layer(input_layer, name="X")

        output_layer = DiehlAndCookNodes(
            n=self.n_neurons,
            traces=True,
            rest=-65.0,
            reset=-60.0,
            thresh=exc_thresh,
            refrac=5,
            tc_decay=100.0,
            tc_trace=20.0,
            theta_plus=theta_plus,
            tc_theta_decay=tc_theta_decay,
        )
        self.add_layer(output_layer, name="Y")

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        input_connection = Connection(
            source=self.layers["X"],
            target=self.layers["Y"],
            w=w,
            update_rule=PostPre,
            nu=nu,
            reduction=reduction,
            wmin=wmin,
            wmax=wmax,
            norm=norm,
        )
        input_connection.update_rule.reduction = torch.sum
        self.add_connection(input_connection, source="X", target="Y")

        w = -self.inh * (
            torch.ones(self.n_neurons, self.n_neurons)
            - torch.diag(torch.ones(self.n_neurons))
        )
        recurrent_connection = Connection(
            source=self.layers["Y"],
            target=self.layers["Y"],
            w=w,
            wmin=-self.inh,
            wmax=0,
        )
        recurrent_connection.update_rule.reduction = torch.sum
        self.add_connection(recurrent_connection, source="Y", target="Y")
        

network = LIFNET(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)
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
# network = DiehlAndCook2015v2(
#     n_inpt=784,
#     n_neurons=n_neurons,
#     exc_thresh=exc,
#     inh=inh,
#     dt=dt,
#     norm=78.4,
#     nu=(1e-4, 1e-2),
#     theta_plus=theta_plus,
#     inpt_shape=(1, 28, 28),)

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


# Directs network to GPU
if gpu:
    network.to("cuda")


# from SortedMNIST import SortedMNIST
# train_dataset = MNIST(
#     PoissonEncoder(time=time, dt=dt),
#     None,
#     root=os.path.join("..", "..","..", "data", "MNIST"),
#     download=True,
#     train=True,
#     transform=transforms.Compose(
#         [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
#     ),
# )
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../../data/MNIST",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}


# last_layer_vm = Monitor(
#     network.layers[f"Ae_{num_layers-1}"], ["v"], time=int(time / dt), device=device
# )
# network.add_monitor(last_layer_vm, name="last_layer_v")

# Set up monitors for spikes and voltages
#will hold the array of axes and ims to be called for each set of 2 layers
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
# voltages[f"Ae_{num_layers-1}"] = Monitor(
#     network.layers[f"Ae_{num_layers-1}"], state_vars=["v"], time=int(time / dt), device=device
# )
# network.add_monitor(voltages[f"Ae_{num_layers-1}"], name=f"%Ae_{num_layers-1}_voltages")

inpt_ims, inpt_axes = None, None

weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# print(spikes.keys())
print("Network layers:", network.layers)

# Train the network.
print("\nBegin training.\n")
start = t()
for connection_name, connection in network.connections.items():
    print(f"Connection Name: {connection_name}")
    
    source_name = getattr(connection.source, "name", None)
    target_name = getattr(connection.target, "name", None)
    
    print(f"Source Layer: {source_name}")
    print(f"Target Layer: {target_name}")
    print(f"Weight Matrix Shape: {connection.w.shape}")
    print("=" * 30)



#############################################################

for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
        train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    for step, batch in enumerate(tqdm(train_dataloader)):
        if (step+1) * batch_size > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time/dt), batch_size, 1, 28, 28)}
        # print(inputs["X"].shape)
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if (step) * batch_size % update_interval == 0 and step > 0:
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

        labels.extend(batch["label"].tolist())

        # Run the network on the input.
        network.run(inputs=inputs, time=time)#, reward=10)

        # last_layer_v = last_layer_vm.get("v")
        # s = spikes[f"Ae_{num_layers-1}"].get("s").permute((1, 0, 2))
        s = spikes[f"Ae"].get("s").permute((1, 0, 2))

        # Add to spikes recording.
        spike_record[
            (step * batch_size) % update_interval: (step * batch_size) % update_interval + s.size(0)
            ] = s

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"][:, 3].view(28, 28)
            inpt = inputs["X"][:, 3].view(time, 784).sum(0).view(28, 28)
            lable = batch["label"][3]
            input_exc_weights = network.connections[("X", "Ae")].w
            square_weights = get_square_weights(
                input_exc_weights.view(784, n_neurons), n_sqrt, 28
            )
            square_assignments = get_square_assignments(assignments, n_sqrt)
            spikes_ = {
                layer: spikes[layer].get("s")[:, 3].contiguous() for layer in spikes
            }
            # voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt_axes, inpt_ims = plot_input(
                image, inpt, label=lable, axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            # perf_ax = plot_performance(
            #     accuracy, x_scale=update_steps * batch_size, ax=perf_ax
            # )
            # voltage_ims, voltage_axes = plot_voltages(
            #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            # )

#             image = batch["image"][0].view(28, 28)
#             # inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
#             # input_exc_weights = network.connections[("X", "Ae")].w
#             # square_weights = get_square_weights(
#             #     input_exc_weights.view(784, n_neurons), n_sqrt, 28
#             # )
#             square_assignments = get_square_assignments(assignments, n_sqrt)
#             spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
#             # voltages = {f"Ae_{num_layers-1}" : last_layer_v}
#             # inpt_axes, inpt_ims = plot_input(
#             #     image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
#             # )
#             #Changes to custom plotting function
#             ###########################################################
#             #CUSTOM PLOTTING - 2 PER PLOT
#             spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
#             spikes_subset = {
#                 layer: spikes_[layer][:, 0:1, :] for layer in spikes_ if layer != "X"
#             }
#             spikes_subset_x = {
#                 "X": spikes_["X"][:, 0:1, :, :, :]
#             }
#             spikes_subset.update(spikes_subset_x)
#             spikes_ = spikes_subset
#             keys = list(spikes_.keys())
#             for i in range(0, len(keys), 2):
#                 # Get two consecutive layers from spikes_
#                 layer1_key = keys[i]
#                 layer2_key = keys[i + 1] if i + 1 < len(keys) else None
                
#                 # Get the spike data for the current layers
#                 layer1_spikes = spikes_[layer1_key]
#                 layer2_spikes = spikes_[layer2_key] if layer2_key else None
#                 if(layer2_spikes == None):
#                     ims[i], axes[i] = plot_spikes(
#                         {layer1_key: layer1_spikes},
#                         ims=ims[i], axes=axes[i]
#                     )
#                 else:
#                     ims[i], axes[i] = plot_spikes(
#                         {layer1_key: layer1_spikes, layer2_key: layer2_spikes},
#                         ims=ims[i], axes=axes[i]
#                     )
# #################################################################################
#             # spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
#             # weights_im = plot_weights(square_weights, im=weights_im)
#             assigns_im = plot_assignments(square_assignments, im=assigns_im)
#             perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
#             # voltage_ims, voltage_axes = plot_voltages(
#             #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
#             # )

#             # plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        # non_negative_indices = torch.nonzero(assignments != -1).squeeze()
        # Print the indices and corresponding values
        # for index in non_negative_indices:
            # print(f"Index: {index}, Assignment: {assignments[index]}")

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..","..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((batch_size, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()
labels = []

pbar = tqdm(total=n_test, position=0, leave=True)
dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=gpu
)
steps = 0
for step, batch in enumerate(dataloader):
    if (step+1) * batch_size > n_test:
        steps = step+1*batch_size
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), batch_size, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    s = spikes[f"Ae"].get("s").permute((1, 0, 2))

    # Add to spikes recording.
    spike_record = s

    # Convert the array of labels into a tensor
    # label_tensor = torch.tensor(batch["label"], device=device)
    labels = batch["label"].tolist()
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
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / steps))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / steps))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

# torch.save(network.state_dict(), 'MultiLayerBatch1Epoch.pth')