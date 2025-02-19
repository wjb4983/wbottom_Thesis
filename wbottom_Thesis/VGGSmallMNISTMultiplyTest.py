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
        self.random_factors = nn.Parameter(torch.randn(3))  # Random factors for multiplication

        # Divide the classifier into separate blocks for possible manipulation
        self.classifier_block1 = nn.Sequential(
            nn.Linear(128 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier_block2 = nn.Sequential(
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
        )
        self.classifier_final = nn.Linear(4096, num_classes)


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)

        # Multiply output of each block with a random factor
        x = self.classifier_block1(x) * self.random_factors[0]
        x = self.classifier_block2(x) * self.random_factors[1]
        x = self.classifier_final(x) * self.random_factors[2]

        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 1  # Assuming input channels is 1 for MNIST
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
    
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
    batch_size = 64
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
    learning_rate = 1e-4
    batch_size = 64
    num_epochs = 10 # You may change number of epochs here. 10 epochs may take up to 10 minutes for training.
    
    # Load pretrain model & you may modify it
    model = VGGSmall(num_classes=10)
    model.to(device)
    
    # Loss and optimizer
    # Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
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
    
    check_accuracy(train_loader, model)
    check_accuracy(val_loader, model)
    
    torch.save(model.state_dict(), 'VGGSmallMNIST.pth')