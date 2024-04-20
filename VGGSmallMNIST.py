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

# This is for the progress bar.
from tqdm.auto import tqdm
import ssl

# VGG11 according to ChatGPT
class VGGSmall(nn.Module):
    def __init__(self, num_classes=10):
        super(VGGSmall, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
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
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    

class VGGSmallEx(nn.Module):
    def __init__(self, num_classes=10, loss_chance=0.0):
        super(VGGSmallEx, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 7 * 7, 4096)
        self.relu5 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout()
        self.fc2 = nn.Linear(4096, 4096)
        self.relu6 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout()
        self.fc3 = nn.Linear(4096, num_classes)
        self.loss_chance=loss_chance

    def forward(self, x):
        x = self.conv1(x)
        x = self.stochastic_activation(x)
        x = self.relu1(x)
        x = self.stochastic_activation(x)
        x = self.conv2(x)
        x = self.stochastic_activation(x)
        x = self.relu2(x)
        x = self.stochastic_activation(x)
        x = self.maxpool1(x)
        
        x = self.conv3(x)
        x = self.stochastic_activation(x)
        x = self.relu3(x)
        x = self.stochastic_activation(x)
        x = self.conv4(x)
        x = self.stochastic_activation(x)
        x = self.relu4(x)
        x = self.stochastic_activation(x)
        x = self.maxpool2(x)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.stochastic_activation(x)
        x = self.relu5(x)
        x = self.stochastic_activation(x)
        x = self.fc2(x)
        x = self.stochastic_activation(x)
        x = self.relu6(x)
        x = self.stochastic_activation(x)
        x = self.fc3(x)
        return x
    def stochastic_activation(self, x):
        mask = torch.rand_like(x) < self.loss_chance  # 5% probability for 0, 95% probability for 1
        return x * (~mask).float()  # Apply mask to zero out 5% of the values
def check_accuracy(loader, model, device, arr):
    # if loader.dataset.train:
        # print("Checking accuracy on training data")
    # else:
        # print("Checking accuracy on test data")

    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(
            f"Got {num_correct} / {num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}"
        )
    arr = np.append(arr, float(num_correct)/float(num_samples))
    model.train()
    
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
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    learning_rate = 1e-4
    batch_size = 256
    num_epochs = 2 # You may change number of epochs here. 10 epochs may take up to 10 minutes for training.
    
    # Load pretrain model & you may modify it
    model = VGGSmallEx(num_classes=10)
    model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
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

    
    check_accuracy(train_loader, model)
    check_accuracy(val_loader, model)
    
    torch.save(model.state_dict(), 'VGGSmallMNIST.pth')