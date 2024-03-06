# Adapted from Deep Learning HW2
# Import necessary packages.
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

# Transforms
my_transforms = transforms.Compose(
    [
     transforms.ToTensor(),
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

# Simple CNN
class CNN(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(CNN, self).__init__()
        #(1pt)TODO: Fill the missing with correct number
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=8, # hint: in_channels of next conv layer
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )

        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html
        #(1pt)TODO: Use a Max Pooling layer with kernel size = 2 and stride =2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
        )
        #(1pt)TODO: Fill the missing with correct number
        self.fc1 = nn.Linear(16 * 7 * 7, num_classes) # hint: feature channels * length * width

    def forward(self, x):
        x = self.conv1(x)                   #(1pt)TODO: Use conv1 layer defined above
        x = F.relu(x)                       #(1pt)TODO: Use relu layer
        x = self.pool(x)                    #(1pt)TODO: Use pooling layer defined above
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc1(x)                     #(1pt)TODO: Use fully connection layer defined above
        return x

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
learning_rate = 1e-4
batch_size = 64
num_epochs = 10 # You may change number of epochs here. 10 epochs may take up to 10 minutes for training.

# Load pretrain model & you may modify it
model = CNN(in_channels=1, num_classes=10)
model.classifier = nn.Sequential(nn.Linear(512, 100), nn.ReLU(), nn.Linear(100, 10))
model.to(device)

# Loss and optimizer
# Reference: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
criterion = nn.CrossEntropyLoss() # (1pt)TODO: Use CrossEntropy as loss criterion
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train Network
for epoch in range(num_epochs):
    losses = []

    # reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
    for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
        # (1pt) TODO: Get data to cuda if possible
        data = data.to(device)
        targets = targets.to(device)

        # (1pt) TODO: forward
        outputs = model(data)
        loss = criterion(outputs, targets)

        losses.append(loss.item())
        # (1pt) TODO: backward
        optimizer.zero_grad()
        loss.backward()

        # (1pt) TODO: gradient descent or adam step
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

