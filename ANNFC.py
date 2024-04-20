import torch
import torch.nn as nn
import torch.nn.functional as F

class FCNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, loss_chance, num_layers):
        super(FCNetwork, self).__init__()
        self.fc_layers = nn.ModuleList()
        self.fc_layers.append(nn.Linear(input_size, hidden_size))
        self.fc_layers.append(nn.ReLU())
        for i in range(num_layers):
            self.fc_layers.append(nn.Linear(hidden_size, hidden_size))
            self.fc_layers.append(nn.ReLU())
        self.softmax = nn.Softmax(dim=1)  # Softmax along the second dimension
        self.loss_chance = loss_chance

    def forward(self, x):
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = layer(x)
            if(layer.__class__.__name__ == "ReLU"):
                x = self.stochastic_activation(x)
        x = self.softmax(x)
        return x

    def stochastic_activation(self, x):
        mask = torch.rand_like(x) < self.loss_chance  # 5% probability for 0, 95% probability for 1
        return x * (~mask).float()  # Apply mask to zero out 5% of the values
