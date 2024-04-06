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
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network import Network

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=50000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=500)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--batch_size", type=int, default=50)
parser.add_argument("--new_model", type=int, default="0") #1 if you want new model, 0 if use pretrained model
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
batch_size=args.batch_size

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
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=theta_plus,
    inpt_shape=(1, 28, 28),
)

if os.path.isfile("diehlcook.pth"):
    print("=======================================\nUsing diehlcook.pth found on your computer\n============================")
    # network.load_state_dict(torch.load('diehlcook.pth'))
else:
    print("=======================================\nCreating new model - saved as diehlcook.pth\n============================")
    #note model created above
# Directs network to GPU
if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..","..", "data", "MNIST"),
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)
#########################################################
# Sort Data in case the program learns better when getting the same data
sorted_train_dataset = []
# for item in train_dataset:
#     # print(item.shape)
#     print(item.keys())

# for data_label_tuple in train_dataset:
#     # Unpack the data and label from the tuple
#     data, label = data_label_tuple
#     sorted_train_dataset.append((data, label))

train_dataset = sorted(train_dataset, key=lambda x: x["label"])
print("s")

# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
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


# Train the network.
print("\nBegin training.\n")
start = t()
for epoch in range(n_epochs):
    labels = []

    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    # Create a dataloader to iterate and batch data
    dataloader = torch.utils.data.DataLoader(
        sorted_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=gpu
        # train_dataset, batch_size=batch_size, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if (step+1) * batch_size > n_train:
            break
        # Get next input sample.
        # inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        inpts = {"X": batch["encoded_image"].reshape(int(time/dt), batch_size, 1, 28, 28)}
        if gpu:
            # inputs = {k: v.cuda() for k, v in inputs.items()}
            inpts = {k: v.cuda() for k, v in inpts.items()}

        if (step) * batch_size % update_interval == 0 and step > 0:
            # print(f"s{step},b{batch_size},u{update_interval}")
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)
            # print(label_tensor.shape)
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
            # print(label_tensor.shape)
            # Compute network accuracy according to available classification strategies.
            accuracy["all"].append(100
                * torch.sum(label_tensor.long() == all_activity_pred).item()
                / len(label_tensor)
            )
            accuracy["proportion"].append(100
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

        # labels.append(batch["label"])
        labels.extend(batch["label"].tolist())
        print(labels)
        # print(f"Step: {step}, Input Shape: {inpts['X'].shape}")

        # Run the network on the input.
        network.run(inputs=inpts, time=time)

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        print(s.shape)
        print(spike_record.shape)
        s_in = spikes["X"].get("s")#.permute((1, 0, 2))
        s_in = s_in.squeeze().reshape(250, 28, 28)
        s_in = s_in.view(250, -1)
        # Add to spikes recording.
        # spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()
        # spike_record[
        #     (step * batch_size)
        #     % update_interval: (step * batch_size % update_interval)
        #                        + s.size(0)
        #     ] = s
        spike_record[
            (step * batch_size) % update_interval: (step * batch_size) % update_interval + s.size(0)
            ] = s
                
        # s_permuted = s_in.permute((2, 1, 0))
                # Create a plot
        plt.figure(figsize=(10, 6))
        
        # Iterate over the dimensions to plot the spikes
        for x in range(s_in.shape[0]):
            for y in range(s_in.shape[1]):
                # Check if there are spikes at this location
                if s_in[x, y].sum() > 0:
                    plt.scatter(x, y, color='black', marker='o')
        
        # Set labels and title
        plt.xlabel('Time Step')
        plt.ylabel('Neuron Index')
        plt.title('Spikes Plot')
        # Optionally plot various simulation information.
        if plot:
            image = batch["image"][0].view(1,28,28).permute(1, 2, 0)
            image = image / image.max()
            inpt = inpts["X"][0][0].view(1,28,28)
            inpt = inpt.squeeze() 
            local_inpy = inpt.reshape(inpt.shape[0], -1)
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            
            # curves, preds = update_curves(
            #         curves, label_tensor, n_classes, spike_record=spike_record, assignments=assignments,
            #         proportions=proportions, ngram_scores=ngram_scores, n=2
            #     )
            # input_exc_weights = network.connections[("X", "Ae")].w
            # square_weights = get_square_weights(
            #     input_exc_weights.view(784, n_neurons), n_sqrt, 28
            # )
            # square_assignments = get_square_assignments(assignments, n_sqrt)

            for key, value in spikes.items():
                # Assuming 'value' is a torch.Tensor representing spikes
                spikes_[key] = value.get("s")[:batch_size, 0]  # Extracting spikes for the first image
            voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
            
            # print(type(spike_record))
            # spikes_ = spike_record[(step * batch_size) % update_interval]["Ae"].get("s")
            
            inpt_axes, inpt_ims = plot_input(
                image, local_inpy, label=batch["label"][0], axes=inpt_axes, ims=inpt_ims
            )
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            # weights_im = plot_weights(square_weights, im=weights_im)
            # assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
            voltage_ims, voltage_axes = plot_voltages(
                voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
            )

            # plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.
        # non_negative_indices = torch.nonzero(assignments != -1).squeeze()
        # Print the indices and corresponding values
        # for index in non_negative_indices:
        #     print(f"Index: {index}, Assignment: {assignments[index]}")
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
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test, position=0, leave=True)
dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=False, num_workers=n_workers, pin_memory=gpu
)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    # inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    inpts = {"X": batch["encoded_image"].reshape(int(time/dt), batch_size, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inpts.items()}

    # Run the network on the input.
    network.run(inputs=inpts, time=time)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()

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
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )

    network.reset_state_variables()  # Reset state variables.
    # pbar.set_description_str("Test progress: ")
    # pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

# torch.save(network.state_dict(), 'diehlcook.pth')