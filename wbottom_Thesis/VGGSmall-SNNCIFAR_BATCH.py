import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm.auto import tqdm
from bindsnet.datasets import CIFAR100
from VGGSmallCifar100 import VGGSmall

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights

class TorchvisionDatasetWrapper:
    def __init__(self, dataset_name):
        self.dataset_name = dataset_name


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=50000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128*25)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=1000)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", default=False)
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--batch_size", type=int, default=25)
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
batch_size = args.batch_size

# Sets up GPU usage
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
torch.manual_seed(seed)

# Determines the number of workers to use
if n_workers == -1:
    n_workers = os.cpu_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

import torch.nn as nn

class VGGSmallFeatureExtractor(nn.Module):
    def __init__(self, num_classes=100):
        super(VGGSmallFeatureExtractor, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 'M', 256, 'M'])#, 512, 512, 'M', 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(256 *4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
         )

    def forward(self, x):
        x = self.features(x)
        return x
    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

VGGSmallFE = VGGSmallFeatureExtractor()
VGGSmallFE.load_state_dict(torch.load('VGGSmallCifar100.pth', map_location=torch.device(device)))

###POISON ENCODER FOR VGGSMALL OUTPUT
encoder = PoissonEncoder(time=time, dt=dt, intensity=intensity)

# Build network
network = DiehlAndCook2015(
    n_inpt=256 * 4 * 4,  # Adjusted input size for feature extraction in VGGSmall
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=(256*4*4)/10000,#78.4,
    theta_plus=theta_plus,
    inpt_shape=(256,4,4),  # Adjusted input shape for CIFAR-100 images
)
if os.path.isfile("VGGSmall-SNNAugment.pth"):
    print("=======================================\nUsing VGGSmall-SNNAugment(network).pth found on your computer\n============================")
    network.load_state_dict(torch.load('VGGSmall-SNNAugment(network).pth'))
else:
    print("=======================================\nCreating new model - saved as VGGSmall-SNNAugment(network).pth\n============================")

# Directs network to GPU
network.to(device)


train_dataset = CIFAR100(
    # PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "..", "data", "CIFAR100"),
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

# Record spikes during the simulation
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions
n_classes = 100  # CIFAR-100 has 100 classes
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
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

pbar = tqdm(total=n_train, position=0, leave=True)
# Train the network
print("\nBegin training.\n")
start = t()
rstep=0
for epoch in range(n_epochs):
    labels = []
    if (rstep > n_train):
        break
    # if epoch % progress_interval == 0:
        # print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        # start = t()

    # Create a DataLoader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=gpu#n_workers, pin_memory=gpu
    )
    for step, batch in enumerate(dataloader):#tqdm(dataloader)):
        if (step+1) * batch_size > n_train:
            break
        CNN_out = VGGSmallFE(batch["encoded_image"])
        spike_train = encoder(CNN_out)
        inputs = {"X": spike_train.view(int(time / dt), batch_size, 256,4,4).to(device)}
        # if gpu:
        #     inputs = {k: v.cuda() for k, v in inputs.items()}


        if (step) * batch_size % update_interval == 0 and rstep > 0:
            # torch.save(network.state_dict(), 'diehlcookcifar.pth')
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies
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
                " (best)\n\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )

            # Assign labels to excitatory layer neurons
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        # labels.append(batch["label"])
        labels.extend(batch["label"].tolist())
        # print(f"Step: {step}, Input Shape: {inpts['X'].shape}")

        # Run the network on the input.
        network.run(inputs=inputs, time=time)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")
        s = spikes["Ae"].get("s").permute((1, 0, 2))


        spike_record[
            (step * batch_size) % update_interval: (step * batch_size) % update_interval + s.size(0)
            ] = s



        # Optionally plot various simulation information
        if plot:
            # image = individual_image
            image = batch["image"][0].view(3,32,32).permute(1, 2, 0)
            image = image / image.max()
            inpt = inputs["X"][0][0]#.view(256,4,4)
            # inpt = inputs["X"][0].sum(dim=0).view(128, 7, 7)#.view(time, 7*7*128).sum(0).view(7,7,128)
            # input_exc_weights = network.connections[("X", "Ae_0")].w
            # square_weights = get_square_weights(
            #     input_exc_weights.view(3072, n_neurons), n_sqrt, 32
            # )
            square_assignments = get_square_assignments(assignments, 100)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            for key, value in spikes.items():
                # Assuming 'value' is a torch.Tensor representing spikes
                spikes_[key] = value.get("s")[:batch_size, 0]#Extracting spikes for the first image
            # print(spikes["X"])
            # voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            inpt = inpt.squeeze() 
            local_inpy = inpt.reshape(inpt.shape[0], -1)
            
            inpt_axes, inpt_ims = plot_input(
                image, local_inpy, label=batch["label"][0], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            # weights_im = plot_weights(square_weights, im=weights_im)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            # voltage_ims, voltage_axes = plot_voltages(
            #     voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            # )
    
            # plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables
        pbar.set_description_str("Train progress: ")
        pbar.update()
        rstep=rstep+1

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

# Load CIFAR-100 test data
test_dataset = CIFAR100(
    # PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "..", "data", "CIFAR100"),
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)
# Sequence of accuracy estimates
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Test the network
print("\nBegin testing\n")
network.train(mode=False)
rstep=0
start = t()

pbar = tqdm(total=n_test, position=0, leave=True)
for step, batch in enumerate(test_dataset):
    if rstep >= n_test:
        break
    CNN_out = VGGSmallFE(batch["encoded_image"])
    spike_train = encoder(CNN_out)
    inputs = {"X": spike_train.view(int(time / dt), batch_size, 256,4,4).to(device)}

    # Run the network on the input
    network.run(inputs=inputs, time=time)

    # Add to spikes recording
    spike_record[0] = spikes["Ae"].get("s").squeeze()

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables
    pbar.set_description_str("Test progress: ")
    pbar.update()
    rstep=rstep+1

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

# Save the trained network weights
torch.save(network.state_dict(), 'VGGSmall-SNNAugment(network).pth')
