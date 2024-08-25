import torch
import torch.nn as nn
import torch.nn.functional as F

# Adapted from Deep Learning HW2
# Import necessary packages.
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn  # All neural network modules, nn.Linear, nn.Conv2d, BatchNorm, Loss functions
import torch.optim as optim  # For all Optimization algorithms, SGD, Adam, etc.
import torch.nn.functional as F  # All functions that don't have any parameters
from torch.utils.data import (DataLoader,)  # Gives easier dataset managment and creates mini batches
import torchvision  # torch package for vision related things
import torchvision.datasets as datasets  # Has standard datasets we can import in a nice way
import torchvision.transforms as transforms  # Transformations we can perform on our dataset
from PIL import Image
from bindsnet.conversion import ann_to_snn

# This is for the progress bar.
from tqdm.auto import tqdm
import ssl



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
            # if(layer.__class__.__name__ == "ReLU"):
                # x = self.stochastic_activation(x)
        x = self.softmax(x)
        return x

    def stochastic_failure(self, x):
        mask = torch.rand_like(x) < self.loss_chance 
        return x * (~mask).float()  
    
class FCNetworkHC(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, loss_chance, num_layers):
        super(FCNetworkHC, self).__init__()
        
        # Define layers explicitly
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        # self.output_layer = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)  # Softmax along the second dimension
        self.loss_chance = loss_chance

    def forward(self, x):
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer(x)
        x = self.relu(x)
        # x = self.output_layer(x)
        x = self.softmax(x)
        return x

if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Transforms
    parser = argparse.ArgumentParser()
    parser.add_argument("--intensity", type=float, default=128)#Change???
    args = parser.parse_args()
    intensity = args.intensity
    my_transforms = transforms.Compose(
        [
         transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)
         ]
    )
    
    # Download the dataset
    train_dataset = datasets.MNIST(
        root=os.path.join("..", "..","..", "data", "MNIST"), train=True, transform=my_transforms, download=True
    )
    val_dataset = datasets.MNIST(
        root=os.path.join("..", "..","..", "data", "MNIST"), train=False, transform=my_transforms,download=True
    )
    test_dataset = datasets.MNIST(
        root=os.path.join("..", "..","..", "data", "MNIST"), train=False, transform=my_transforms,download=True
    )
    
    # Load Data
    batch_size = 512
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    # Random pick a sample to plot
    sample = next(iter(train_loader))
    imgs, lbls = sample
    
    # Plot the sample
    plt.figure(figsize=(2,2))
    grid = torchvision.utils.make_grid(nrow=20, tensor=imgs[1])
    plt.imshow(np.transpose(grid, axes=(1,2,0)), cmap='gray');
    
    
    
    
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    learning_rate = 1e-5
    batch_size = 512
    num_epochs = 2 # You may change number of epochs here. 10 epochs may take up to 10 minutes for training.
    
    # Load pretrain model & you may modify it
    model = FCNetworkHC(1*28*28, 100, 10, 0.0, 1)
    model.to(device)
    
    # Loss and optimizer
    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(model)
    # Train Network
    for epoch in range(num_epochs):
        losses = []
    
        # reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
    
            outputs = model(data)
            loss = criterion(outputs, targets)
    
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
    
            optimizer.step()
    
        print(f"Loss at epoch {epoch + 1} is {sum(losses)/len(losses):.5f}\n")
    
    # Check accuracy on training & test to see how good our model
    def check_accuracy(loader, model):
        if loader.dataset.train:
            print("Checking accuracy on training data")
        else:
            print("Checking accuracy on test data")
    
        num_correct = 0
        num_samples = 0
        model.eval()
    
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device=device)
                y = y.to(device=device)
    
                scores = model(x)
                _, predictions = scores.max(1)
                num_correct += (predictions == y).sum()
                num_samples += predictions.size(0)
    
            print(
                f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
            )
    
        model.train()
    
    # check_accuracy(train_loader, model)
    # check_accuracy(val_loader, model)
    model.to("cpu")
    # snn = ann_to_snn(model, (1,28,28))
    # print(snn)
    torch.save(model.state_dict(), 'presnn_conversionFC.pth')