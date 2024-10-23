import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models

# Set the device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-100 dataset normalization and transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize images for ResNet models
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))  # CIFAR-100 mean and std
])

# Download CIFAR-100 dataset
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, transform=transform, download=True)

# DataLoader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# Load a pretrained smaller CNN model like resnet18
model = models.resnet18(pretrained=True)

# Modify the final layer to fit CIFAR-100 classes
model.fc = nn.Linear(model.fc.in_features, 100)

# Move model to device
model = model.to(device)

# Save the model to the current directory
torch.save(model.state_dict(), './resnet18_cifar100.pth')

print("Model downloaded and saved to current directory as resnet18_cifar100.pth")
