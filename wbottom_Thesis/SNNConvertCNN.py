import torch
from bindsnet.conversion import ann_to_snn
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
from bindsnet.network.monitors import Monitor
from time import time as t_
import pandas as pd
import os

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)


percentile = 99.9
random_seed = 0
torch.manual_seed(random_seed)

batch_size = 32
time = 100

ANN_accuracy = 0
SNN_accuracy = 0

if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    device = torch.device('cuda')
    print("Cuda is available")
else:
    device = torch.device('cpu')
    print("Cuda is not available")


class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.features = self._make_layers([4])#, 128, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(144, num_classes, bias=False),
            # nn.ReLU(True),
            # nn.Linear(100, num_classes, bias=False),
            # nn.ReLU(True),
            # nn.Linear(4096, num_classes, bias=False),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1  # Change input channels to 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=7, padding=1, stride = 4)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    def normalize_weights(model):
        with torch.no_grad():
            model.features[0].weight.data /= 1.0
            # model.fc2.weight.data /= 10.0

        # self.fc1 = nn.Linear(28 * 28, 1000)
        # self.fc2 = nn.Linear(1000, 10)
        # self.reg_strength = reg_strength

    # def forward(self, x):
    #     x = x.view(-1, 28*28)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)

    #     # L2 regularization term
    #     l2_reg = self.reg_strength * (torch.norm(self.fc1.weight) + torch.norm(self.fc2.weight))

    #     return F.log_softmax(x, dim=1) - l2_reg  # Subtract regularization term from output

class FlattenTransform:
    def __call__(self, x):
        return x.view(-1, 28*28)

train_dataset = datasets.MNIST('./data',
                               train=True,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor()]))

train_dataset2 = datasets.MNIST('./data',
                               train=False,
                               download=True,
                               transform=transforms.Compose([transforms.ToTensor(),FlattenTransform()]))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           shuffle=True, batch_size=1, generator=torch.Generator(device=device))

train_loader2 = torch.utils.data.DataLoader(dataset=train_dataset2,
                                           shuffle=True, generator=torch.Generator(device=device))


# for d, target in train_loader:
#     data = d.to(device)



model = Net()

model.load_state_dict(torch.load("presnn_conversion.pth"))
model.normalize_weights()
# model = torch.load('trained_model.pt')

print()
print('Converting ANN to SNN...')

data=None
SNN = ann_to_snn(model, input_shape=(1,28,28), data=None, percentile=percentile)

print(SNN)

SNN.add_monitor(
    Monitor(SNN.layers['3'], state_vars=['v'], time=time), name='3'
)
# for conn_name, conn in SNN.connections.items():
#     print(f'Weights of {conn_name}:')
#     print(conn.w)

# Print biases and injected currents of the layers (if applicable)
# for layer_name, layer in SNN.layers.items():
#     print(f'Layer: {layer_name}')
    
#     if hasattr(layer, 'v'):
#         print(f'Voltage (v): {layer.v}')
    
#     if hasattr(layer, 'i'):
#         print(f'Injected current (i): {layer.i}')
    
#     if hasattr(layer, 'bias'):
#         print(f'Bias: {layer.bias}')

SNN.to(device)
i=0
for connection in SNN.connections.values():
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
    i=i+1

correct = 0

voltage_ims = None
voltage_axes= None
ims = [None] * len(set(SNN.layers))
axes = [None] * len(set(SNN.layers))
spike_ims, spike_axes = None, None
spikes = {}
for layer in set(SNN.layers):
    spikes[layer] = Monitor(
        SNN.layers[layer], state_vars=["s"], time=int(time / 1.0), device=device
    )
    SNN.add_monitor(spikes[layer], name="%s_spikes" % layer)

def validate():
    global ANN_accuracy
    model.eval()
    val_loss, correct = 0, 0
    for data, target in train_loader2:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()


    ANN_accuracy = 100. * correct.to(torch.float32) / len(train_loader2.dataset)

    print("ANN accuracy:", ANN_accuracy)

# validate()


start = t_()
for index, (data, target) in enumerate(train_loader):
    print('sample ', index+1, 'elapsed', t_() - start)
    start = t_()

    data = data.to(device)
    data = data.view(-1, 28,28)
    inpts = {'Input': data.repeat(time,1, 1,1)}
    print(inpts["Input"].shape)
    SNN.run(inputs=inpts, time=time)
    s = {layer: SNN.monitors[f'{layer}_spikes'].get('s') for layer in SNN.layers}
    voltages = {layer: SNN.monitors[layer].get('v') for layer in ['3'] if not layer == 'Input'}
    # pred = torch.argmax(voltages['3'].sum(1))
    # summed_voltages = voltages['3'].sum(0)
    summed_spikes=s['3'].sum(0)
    # pred = torch.argmax(summed_voltages, dim=1).to(device)
    pred = torch.argmax(summed_spikes, dim=1).to(device)
    print(pred, target)
    # correct += pred.eq(target.data.to(device)).cpu().sum()
    correct += pred.eq(target).sum().item()
    accuracy = 100.0 * float(correct) / (index + 1)
    print(correct)
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
    SNN.reset_state_variables()

SNN_accuracy = 100.0 * float(correct) / len(train_loader2.dataset)


print("accuracy:, ", SNN_accuracy)

df = pd.DataFrame({"ANN accuracy":[ANN_accuracy],
                   "SNN accuracy": [SNN_accuracy]})

df.to_csv("accuracy_hidden_1.csv")