import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn  
import torch.optim as optim  
import torch.nn.functional as F 
from torch.utils.data import (DataLoader,) 
import torchvision  
import torchvision.datasets as datasets  
import torchvision.transforms as transforms  
from PIL import Image
from tqdm.auto import tqdm
import ssl
from VGGSmallMNIST import check_accuracy, VGGSmall, VGGSmallEx
from ANNFC import FCNetwork
from VGG16 import VGG16Ex, EncoderDecoder
import torchvision.models as models
from VGG11Cifar100 import VGG11Ex
import pandas as pd
dataset = "cifar100"
encoder = False


if __name__ == '__main__':
    ssl._create_default_https_context = ssl._create_unverified_context
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.manual_seed_all(0)
    torch.manual_seed(0)
    np.random.seed(0)
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
    train_dataset = None
    if(dataset == "mnist"):
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
    if(dataset == "cifar10"):
    # Download the dataset
        train_dataset = datasets.CIFAR10(
            root=os.path.join("..", "..","..", "data", "CIFAR10"), train=True, transform=my_transforms, download=True
        )
        val_dataset = datasets.CIFAR10(
            root=os.path.join("..", "..","..", "data", "CIFAR10"), train=False, transform=my_transforms,download=True
        )
        test_dataset = datasets.CIFAR10(
            root=os.path.join("..", "..","..", "data", "CIFAR10"), train=False, transform=my_transforms,download=True
        )
    if(dataset == "cifar100"):
    # Download the dataset
        print("cifar100")
        train_dataset = datasets.CIFAR100(
            root=os.path.join("..", "..","..", "data", "CIFAR100"), train=True, transform=my_transforms, download=True
        )
        val_dataset = datasets.CIFAR100(
            root=os.path.join("..", "..","..", "data", "CIFAR100"), train=False, transform=my_transforms,download=True
        )
        test_dataset = datasets.CIFAR100(
            root=os.path.join("..", "..","..", "data", "CIFAR100"), train=False, transform=my_transforms,download=True
        )

    # if device=="gpu" and torch.cuda.is_available():
        # torch.cuda.manual_seed_all(0)

    # Load Data
    batch_size = 64
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    # Set device

    
    # Hyperparameters
    learning_rate = 4e-5
    num_epochs = 20
    
    
    #create model array
    # models = {}
    # model_o = VGGSmallEx(num_classes=10)
    # models.add(model)
    # model_o = FCNetwork(3*32*32, 200, 10, 0.0, 3) 
    # models.add(model)
    
    # model.to(device)

    model_o = VGG16Ex(num_classes=100)
    # model_o = EncoderDecoder(num_classes=10)
    # model_o = VGG11Ex(num_classes=10)
    # Loss and optimizer

    import copy
    print(model_o)
    # Train Network
    first_image = None
    # p = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    yes_yes = np.array([])
    yes_no = np.array([])
    no_yes = np.array([])
    while learning_rate < 1e-2:
        print("learning rate = ", learning_rate)
        model = copy.deepcopy(model_o)
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        if(encoder == True):
            criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            losses = []

            # reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
            for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)): #
                data = data.to(device)
                targets = targets.to(device)
        
                outputs = model(data)
                if(encoder == True):
                    loss = criterion(outputs, data)
                    first_image = data[0].cpu().detach()
                else:
                    loss = criterion(outputs, targets)
                losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        
            print(f"Loss at epoch {epoch + 1} is {sum(losses)/len(losses):.5f}\n")
            if(encoder == True):
        # Plot the original image
                im = copy.deepcopy(first_image)
                im1 = copy.deepcopy(first_image)
                plt.figure(figsize=(8, 4))
                plt.subplot(1, 2, 1)
                plt.imshow(np.transpose(im1/255, (1, 2, 0)))
                plt.title('Original Image')
                plt.axis('off')
        
                # Pass the image through the encoder
                im = model(im.unsqueeze(0).to(device))
                im = im.squeeze(0).cpu().detach()
        
                # Plot the reconstructed image
                plt.subplot(1, 2, 2)
                plt.imshow(np.transpose(im/255, (1, 2, 0)))
                plt.title('Reconstructed Image')
                plt.axis('off')
        
                plt.show()
            else:
                empty = []
                check_accuracy(val_loader, model, device, empty)
            if(epoch %10 == 0):
                learning_rate = learning_rate/2
        learning_rate = learning_rate*4
        print("="*30)
    model = copy.deepcopy(model_o)
    model.loss_chance = 0.0
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    if(encoder == True):
        criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        losses = []
        
        # reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
        for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = data.to(device)
            targets = targets.to(device)
    
            outputs = model(data)
            if(encoder == True):
                loss = criterion(outputs, data)
            else:
                loss = criterion(outputs, targets)
    
            losses.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
    
            optimizer.step()
        print(f"Loss at epoch {epoch + 1} is {sum(losses)/len(losses):.5f}\n")
    print("x = ", 0.0)
    emp = []
    check_accuracy(train_loader, model, device, emp)
    # check_accuracy(val_loader, model, device)
    for x in p:
        model.loss_chance = x
        print("x: ",x)
        no_yes = check_accuracy(train_loader, model, device, no_yes)
        # check_accuracy(val_loader, model, device)
        print("="*30)
    arr = np.column_stack((p, yes_yes, yes_no, no_yes))
    arr = arr.reshape(-1,4)
    df = pd.DataFrame(arr, columns=["%", "yes_yes", "yes_no", "no_yes"])
    df.to_csv(f'FC_ep_{num_epochs}_lr_{learning_rate}_ds_{dataset}_bs_{batch_size}.csv', index=False)
    # Check accuracy on training & test to see how good our model

    
    # check_accuracy(train_loader, model, device)
    # check_accuracy(val_loader, model, device)
    # model.loss_chance = 0.05
    # check_accuracy(train_loader, model, device)
    # check_accuracy(val_loader, model, device)
    # model.loss_chance = 0.10
    # check_accuracy(train_loader, model, device)
    # check_accuracy(val_loader, model, device)
    # model.loss_chance = 0.15
    # check_accuracy(train_loader, model, device)
    # check_accuracy(val_loader, model, device)
    # model.loss_chance = 0.20
    # check_accuracy(train_loader, model, device)
    # check_accuracy(val_loader, model, device)
    # model.loss_chance = 0.25
    # check_accuracy(train_loader, model, device)
    # check_accuracy(val_loader, model, device)
    
    
