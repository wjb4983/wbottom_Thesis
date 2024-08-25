import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from bindsnet.conversion import ann_to_snn
from bindsnet.network import Network
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes

# Define a simple ANN model
class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 100)
        self.fc2 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)

# Create and train the ANN model
ann_model = ANN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(ann_model.parameters(), lr=0.01)

for epoch in range(1):  # Train for 5 epochs
    for data, target in train_loader:
        optimizer.zero_grad()
        output = ann_model(data.view(-1, 28 * 28))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# Convert the ANN model to SNN
snn_model = ann_to_snn(ann_model, input_shape=(1, 28, 28), data=data, percentile=99)

# Load test dataset
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Create monitors for spikes and voltages
# snn_model.add_monitor(Monitor(snn_model.layers['fc2'], state_vars=['s', 'v']))
if '2' in snn_model.layers:
    # Add monitor for '1' layer to monitor spikes and voltages
    snn_model.add_monitor(Monitor(snn_model.layers['2'], state_vars=['s', 'v']), name='2_monitor')
else:
    print("'1' layer not found in the SNN model.")

# Test the SNN model only if '1' layer is present
# if '1' in snn_model.layers:
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         snn_model.eval()  # Set SNN model to evaluation mode
#         snn_output = snn_model(images.view(-1, 1, 28, 28))  # Feed input images to SNN model
#         _, predicted = torch.max(snn_output.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the converted SNN on the test images: %d %%' % (100 * correct / total))

#     # Plot spikes and voltages if '1' layer is present
#     if '1' in snn_model.monitors:
#         plot_spikes({'1': snn_model.monitors['1'].get('s')}, figsize=(10, 5))
#     else:
#         print("Spikes monitor for '1' layer not found.")
# else:
#     print("Cannot test SNN model as '1' layer is not present.")

# Test the SNN model
correct = 0
total = 0
print(snn_model)
print("testing")
for images, labels in test_loader:
    # snn_model.eval()  # Set SNN model to evaluation mode
    # snn_output = snn_model(images.view(-1, 1, 28, 28))  # Feed input images to SNN model
    print(images.shape)
    inpts = {'Input': images.view(1000,1, 1, 28, 28)}
    snn_model.run(inputs=inpts, time=100)
    x = snn_model.monitors['2_monitor'].get('s')
    # print(x.shape)
    spike_counts = x.sum(1)  # Sum spikes over time
    # print(spike_counts.shape)
    _, predicted = torch.max(spike_counts, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the converted SNN on the test images: %d %%' % (100 * correct / total))

# Plot spikes and voltages
plot_spikes({'fc2': snn_model.monitors['fc2'].get('s')}, figsize=(10, 5))
