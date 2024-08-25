import torch
import torch.optim as optim
import torch.nn.functional as F
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.learning import MSTDP
from bindsnet.evaluation import all_activity
from bindsnet.learning import NoOp

seed=0
batch_size = 1
n_workers = 0
gpu = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
n_epochs = 1

# Define the spiking neural network
network = Network()
input_layer = Input(n=784)
output_layer = LIFNodes(n=10)
network.add_layer(input_layer, name="input")
network.add_layer(output_layer, name="output")
connection = Connection(
    learning_rule=NoOp,
    source=input_layer,
    target=output_layer,
    wmin=0,
    wmax=1,
)
network.add_connection(connection, source="input", target="output")

from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
intensity = 128
time = 250
dt = 1.0
from torchvision import transforms
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../../data/MNIST",
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)


# Define surrogate gradient function
def surrogate_gradient(x):
    return torch.sign(x)

# Define learning rule (e-STDP with surrogate gradient)
learning_rule = MSTDP(connection, lr=0.01, interaction_matrix=torch.eye(784), update_rule="hebbian", surrogate=surrogate_gradient)

# Define loss function and optimizer
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(network.parameters(), lr=0.001)





spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)
        
n_neurons = 10
n_classes = 10
update_interval = batch_size *5
spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)
assignments = -torch.ones(n_neurons, device=device)
network.to(device)
# Training loop
for epoch in range(n_epochs):
    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=gpu,
    )

    for step, batch in enumerate(train_dataloader):
        inputs, labels = batch["encoded_image"].to(device), batch["label"].to(device)

        network.run(inputs={"input": inputs}, time=time)  # Run the network for a fixed duration

        # Get spike counts from output layer
        s = spikes["output"].get("s").permute((1, 0, 2))#.sum(dim=1)
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s
        optimizer.zero_grad()
        all_activity = all_activity(
            spikes=s, assignments=assignments, n_labels=n_classes
        )
        # print(all_activity.shape)
        # Compute loss and perform backpropagation through time
        print(all_activity)
        print(labels)
        all_activity = all_activity.float().requires_grad_(True)
        loss = loss_fn(all_activity, labels.float())
        loss.backward()
        optimizer.step()

        # Update weights using the learning rule
        # learning_rule(connection)
        learning_rule.update()

        network.reset_state_variables()  # Reset state variables for the next iteration
