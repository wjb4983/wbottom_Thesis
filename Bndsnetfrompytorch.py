import sys
sys.path.append('C:\\Users\\wbott')
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import matplotlib.pyplot as plt
import random
import os
import argparse
from time import time as t
from torchvision import transforms
from tqdm import tqdm
from typing import Iterable, List, Optional, Sequence, Tuple, Union
from bindsnet.learning import PostPre

#from https://github.com/BindsNET/bindsnet/blob/master/examples/mnist/eth_mnist.py
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
from bindsnet.network.nodes import DiehlAndCookNodes, Input, LIFNodes, IFNodes#, Softmax
from bindsnet.network.topology import Connection

verbose = False

class CustomHidden(nn.Module):
    def __init__(self, input_size, num_hidden, activation='sigmoid'):
        super(CustomHidden, self).__init__()
        self.units = num_hidden
        self.kernel_regularizer = nn.Linear(input_size, num_hidden, bias=False)
        self.bias = nn.Parameter(torch.rand(num_hidden))
        self.activation = getattr(torch.nn.functional, activation)

    def forward(self, inputs):
        return self.activation(torch.mm(inputs, self.kernel_regularizer.weight.T) - self.bias)

# Singular node of the tree
# Provides most of the methods for SEQUENTIAL TRAINING
# In order to do simultaneous and/or with reservoir, need special special classifier
# So that we can do backprop the right way
class VascularTreeNode:
    def __init__(self, num_children, is_leaf, energy, is_head):
        # Weights are random at first - this is ok
        # I keep too many variables here - change some to global or somethign later
        self.num_children = num_children
        self.is_leaf = is_leaf
        unnormalized_weights = np.random.rand(num_children)
        self.weights = unnormalized_weights / np.sum(unnormalized_weights)
        self.children = []
        self.energy = energy;
        self.gradient = 0;
        self.alpha = .1#/50
        self.is_head = is_head

    def forward(self, energy):
        # Simulate forward propogation
        # Just calculates what the energy values should be for leafs & intermediate nodes
        if self.is_leaf:
            # print(energy)
            self.energy =np.clip(energy, 0, 2)
            return np.array([self.energy])
        else:
            self.energy = energy
            # print(energy)
            child_energies = self.weights * energy
            return np.concatenate([child.forward(child_energy) for child, child_energy in zip(self.children, child_energies)])
        
        
    def backprop(self, train_bias):
        # Currently only uses gradient between hidden node's bias and calculated 
        # what its new bias would be given energy forward prop
        my_energy = self.getenergies()                     #gets array of leaf energies (EL)
        gradient = ((1-train_bias)-my_energy) * self.alpha #DELTA = EN-EL
        # energy_gradient = 1 + gradient                     #MATH SAYS TO ADD
        if verbose:
            # print("my energy",my_energy)
            print("grad",gradient)
            # print("my",my_bias)
            # print("train",train_bias)
        self.setgradient(gradient)    #set gradient
        for i in range(self.getdepth()):
            self.setenergies(i)                   #then update energy that the node I am at currently distributes down before normalization
        self.updateweights()             

    def updateweights(self):
        # Just updates the weights based on backprop gradients - not ture anymore
        # Intermediate nodes gradient = average of 
        if self.is_leaf:
            return self.energy
        else:
            child_grad = np.array([child.updateweights() for child in self.children])
            child_grad = child_grad.flatten()
            # print(self.weights.size)
            # print(child_grad.size)
            new_grad = np.average(child_grad)
            # print(child_grad)
            if verbose:
                print("before weight",self.weights)
            self.weights = child_grad/np.sum(child_grad)
            if verbose:
                print("after weight",self.weights)

            # This might do something?
            # This is supposed to make the nodes coming from the head node have
            # Custom weights proportional to the energy taken
            # so the weights can add up to < head.energy
            # However once backprop is done, this never happens
            if self.is_head:
                total_used_energy = np.sum(self.getenergies())
                self.weights = self.weights*(total_used_energy/self.energy)
            return new_grad
            
    #sets gradient at leaf nodes
    def setgradient(self, grad):
        # Takes gradient of each leaf calculated and puts it in the right leaf
        # Doesn't give gradient for the intermediate nodes
        if self.is_leaf:
            self.gradient = grad
        else:
            # print(grad)
            # print(self.num_children)
            childgrad = np.split(grad, self.num_children)
            for child, child_gradient in zip(self.children, childgrad):
                child.setgradient(child_gradient)
            
    #gets sum of all energies at leaf
    def checksum(self):
        if self.is_leaf:
            return np.abs(self.energy)
        else:
            return np.sum([child.checksum() for child in self.children])
        
    #gets array of all energies at leaf
    def getenergies(self):
        if self.is_leaf:
            return np.array([self.energy])
        else:
            return np.concatenate([child.getenergies() for child in self.children])
        
    #Updates energies at leaf using the gradient
    def setenergies(self, time):
        if self.is_leaf:
            if time == 0:
                if verbose:
                    print("before",self.energy,"after",self.gradient)
                self.energy = np.clip(self.energy+self.gradient, 0, 2)
            return self.energy
        else:
            self.energy = np.sum([child.setenergies(time) for child in self.children])
            return self.energy
        
    def getdepth(self):
        if self.is_leaf:
            return 0
        else:
            return 1 + self.children[0].getdepth()
            

class VascularTree:
    def __init__(self, numhidden, num_children_per_node):
        self.root = VascularTreeNode(num_children_per_node, False,100, True)
        num_layers = math.ceil(math.log(numhidden,num_children_per_node))
        def add_layer(parent_node, depth):
            if depth < num_layers - 1:
                parent_node.children = [VascularTreeNode(num_children_per_node, False,0, False) for _ in range(num_children_per_node)]
                for child in parent_node.children:
                    add_layer(child, depth + 1)
            else:
                parent_node.children = [VascularTreeNode(0, True,0, False) for _ in range(num_children_per_node)]

        add_layer(self.root, 0)


def reset_random_seeds(n):
    torch.manual_seed(n)
    np.random.seed(n)
    random.seed(n)

n=41
reset_random_seeds(n)




parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=n)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=1000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")   
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
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


# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        
        
# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity



# input_layer = Input(n=input_size, traces=True, tc_trace=20.0)
# lif_layer = LIFNodes(n=n_neurons)
# #softmax_layer = Softmax(n=n_neurons)
# output_layer = IFNodes(n=output_size)

# #https://www.frontiersin.org/files/Articles/409297/fninf-12-00089-HTML/image_m/fninf-12-00089-g005.jpg

# network.add_layer(input_layer, name="input_layer")
# network.add_layer(lif_layer, name="lif_layer")
# network.add_layer(output_layer, name="output_layer")

# # Connect layers.
# input_connection = Connection(input_layer,lif_layer,norm=150, wmin=-1,wmax=1)
# output_connection = Connection(lif_layer,output_layer,norm=150, wmin=-1,wmax=1)
# network.add_connection(input_connection, source="input_layer", target="lif_layer")
# network.add_connection(output_connection, source="lif_layer", target="output_layer")

class mySNN(Network):
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
        self.add_connection(recurrent_connection, source="Y", target="Y")
        
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
        
# Build network.
network = mySNN(n_inpt=784, inpt_shape=(1,28,28))
if os.path.isfile("bindsnet_model.pth"):
    print("=======================================\nUsing pth found on your computer\n============================")
    network.load_state_dict(torch.load('bindsnet_model.pth'))
else:
    print("=======================================\nCreating new model\n============================")

# Directs network to GPU
if gpu:
    network.to("cuda")



# Record spikes during the simulation.
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Neuron assignments and spike proportions.
n_classes = 10
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# # Voltage recording for excitatory and inhibitory layers.
# exc_voltage_monitor = Monitor(
#     network.layers["X"], ["v"], time=int(time / dt), device=device
# )
inh_voltage_monitor = Monitor(
    network.layers["Y"], ["v"], time=int(time / dt), device=device
)
# network.add_monitor(exc_voltage_monitor, name="exc_voltage")
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
        train_dataset, batch_size=1, shuffle=True, num_workers=n_workers, pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(dataloader)):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_interval == 0 and step > 0:
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

        labels.append(batch["label"])

        # Run the network on the input.
        network.run(inputs=inputs, time=time)

        # # Get voltage recording.
        # exc_voltages = exc_voltage_monitor.get("v")
        # inh_voltages = inh_voltage_monitor.get("v")

        # Add to spikes recording.
        # spike_record[step % update_interval] = spikes["Ae"].get("s").squeeze()

        # # Optionally plot various simulation information.
        # if plot:
        #     image = batch["image"].view(28, 28)
        #     inpt = inputs["X"].view(time, 784).sum(0).view(28, 28)
        #     input_exc_weights = network.connections[("X", "Ae")].w
        #     square_weights = get_square_weights(
        #         input_exc_weights.view(784, n_neurons), n_sqrt, 28
        #     )
        #     square_assignments = get_square_assignments(assignments, n_sqrt)
        #     spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
        #     # voltages = {"Ae": exc_voltages, "Ai": inh_voltages}
        #     inpt_axes, inpt_ims = plot_input(
        #         image, inpt, label=batch["label"], axes=inpt_axes, ims=inpt_ims
        #     )
        #     spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
        #     weights_im = plot_weights(square_weights, im=weights_im)
        #     assigns_im = plot_assignments(square_assignments, im=assigns_im)
        #     perf_ax = plot_performance(accuracy, x_scale=update_interval, ax=perf_ax)
        #     voltage_ims, voltage_axes = plot_voltages(
        #         voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
        #     )

        #     plt.pause(1e-8)

        network.reset_state_variables()  # Reset state variables.

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

pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    # spike_record[0] = spikes["Ae"].get("s").squeeze()

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
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")

torch.save(network.state_dict(), 'bindsnet_model.pth')

# For our runs with everything going:
# This will be set up as [numhidden, nunm_children_per_node]
# We will run this loop on all 3 datasets
loop_iter = [[4,2], [4,4], [16,2], [16,4], [32, 2], [64, 2], [128, 2], [256,2], [512,2]]

#set up vascular tree w/ params
# The number of hidden nodes that the classifier will have
# Note it will only have one layer for now
numhidden = 16

num_children_per_node = 2

vascular_tree = VascularTree(numhidden, num_children_per_node)

####################################################################
##need to train classifier to extract biases from:::
    ################################################################
    
# Load Iris dataset
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train_IRIS, X_test_IRIS, y_train_IRIS, y_test_IRIS = train_test_split(X, y, test_size=0.2, random_state=40)

# Standardize the features
scaler = StandardScaler()
X_train_IRIS = scaler.fit_transform(X_train_IRIS)
X_test_IRIS = scaler.transform(X_test_IRIS)


# Create MLP without any vascular stuff and run tests
input_size = X_train_IRIS.shape[1]
output_size = len(set(y_train_IRIS))

# Extract biases
hidden_biases = network.layers[0].bias.detach().numpy()
sum_biases = np.sum(1 - hidden_biases)
if verbose:
    print("Original biases:", hidden_biases)
    print("Sum of original biases:", sum_biases)

# Training the vascular tree
for epoch in range(66):
    if verbose:
        print(f"=============EPOCH {epoch}====================")
    tree_output = vascular_tree.root.forward(sum_biases)
    tree_output = np.maximum(0, tree_output)
    tree_output = np.minimum(2, tree_output)
    vascular_tree.root.backprop(hidden_biases)
    if verbose:
        print("Average gradient:", np.sum(np.abs(hidden_biases - (1 - tree_output))) / numhidden)

if verbose:
    print("Checksum:", vascular_tree.root.checksum())

# Normalize energy -> bias
tree_output = 1 - tree_output
tree_output[tree_output < -1] = -1
if verbose:
    print("Tree output values:", tree_output)
    print("Gradient:", hidden_biases - tree_output)

# Rerun classifier with new biases
network.layers[0].bias.data = torch.tensor(tree_output, dtype=torch.float32)

# Evaluate on the test set with updated biases
with torch.no_grad():
    inputs_test = torch.tensor(X_test_IRIS, dtype=torch.float32)
    labels_test = torch.tensor(y_test_IRIS, dtype=torch.long)
    outputs_test = network(inputs_test)
    _, predicted = torch.max(outputs_test, 1)
    accuracy = (predicted == labels_test).sum().item() / len(labels_test)
    print(f"Accuracy on the test set with energy: {accuracy}")
