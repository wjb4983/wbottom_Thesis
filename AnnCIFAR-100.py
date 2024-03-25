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
ssl._create_default_https_context = ssl._create_unverified_context

# Transforms
parser = argparse.ArgumentParser()
parser.add_argument("--intensity", type=float, default=128)
args = parser.parse_args()
intensity = args.intensity
my_transforms = transforms.Compose(
    [
     transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)
     ]
)

# Download the dataset
train_dataset = datasets.CIFAR100(
    root=os.path.join("..", "..","..", "data", "CIFAR100"), train=True, transform=my_transforms, download=True
)
val_dataset = datasets.CIFAR100(
    root=os.path.join("..", "..","..", "data", "CIFAR100"), train=False, transform=my_transforms,download=True
)
test_dataset = datasets.CIFAR100(
    root=os.path.join("..", "..","..", "data", "CIFAR100"), train=False, transform=my_transforms,download=True
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

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=100, p_dropout=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv6 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.dropout1 = nn.Dropout(p=p_dropout)
        # Adjust the size here based on the actual output from the last conv layer
        self.fc1 = nn.Linear(256 * 8 * 8, 100)
        self.dropout2 = nn.Dropout(p=p_dropout)
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.pool2(F.relu(self.conv6(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout1(F.relu(self.fc1(x)))
        x = self.fc2(self.dropout2(x))
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 1e-4
batch_size = 64
num_epochs = 10 # You may change number of epochs here. 10 epochs may take up to 10 minutes for training.

# Load pretrain model & you may modify it
model = CNN(in_channels=3, num_classes=100, p_dropout=0.5)
model.classifier = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 10))
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

