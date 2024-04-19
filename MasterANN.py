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
dataset = "cifar10"
encoder = True


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # if device=="gpu" and torch.cuda.is_available():
        # torch.cuda.manual_seed_all(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    # Load Data
    batch_size = 32
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)
    
    # Set device

    
    # Hyperparameters
    learning_rate = 1e-3
    num_epochs = 3
    
    
    #create model array
    # models = {}
    # model_o = VGGSmallEx(num_classes=10)
    # models.add(model)
    # model_o = FCNetwork(3*32*32, [500, 500, 500, 500, 500], 10, 0.0) 
    # models.add(model)
    
    # model.to(device)

    # model_o = VGG16Ex(num_classes=100)
    model_o = EncoderDecoder(num_classes=10)
    # Loss and optimizer

    import copy
    # Train Network
    first_image = None
    p = [0.1]#0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for x in p:
        model = copy.deepcopy(model_o)
        model.loss_chance = x
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        if(encoder == True):
            criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        for epoch in range(num_epochs):
            losses = []
            # progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
            # reference : https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
            for batch_idx, (data, targets) in tqdm(enumerate(train_loader), total=len(train_loader)):
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
                # plt.figure(figsize=(4,4))
                # x = outputs.detach().cpu().numpy()
                # # print(x[0])
                # plt.imshow(np.transpose(x[0]/255, (1, 2, 0)))
                # plt.title('Reconstructed Image')
                # plt.axis('off')
        
                optimizer.step()
                # progress_bar.update(batch_size)
        
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
                check_accuracy(val_loader, model, device)
        print("x = ", x, "testing perf where train=true test=true")
        if(encoder == False):
            check_accuracy(train_loader, model, device)
            check_accuracy(val_loader, model, device)
            model.loss_chance = 0.0
            print("x = 0.0 now - testing performance with train=true test=false")
            check_accuracy(train_loader, model, device)
            check_accuracy(val_loader, model, device)
        else:
            im = copy.deepcopy(first_image)
            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(np.transpose(first_image/255, (1, 2, 0)))
            plt.title('Original Image')
            plt.axis('off')
    
            # Pass the image through the encoder
            im1 = model(im.unsqueeze(0).to(device))
            im1 = im1.squeeze(0).cpu().detach()
    
            # Plot the reconstructed image
            plt.subplot(1, 3, 2)
            plt.imshow(np.transpose(im1/255, (1, 2, 0)))
            plt.title(f'Reconstructed Image_{x}')
            plt.axis('off')
    
            
            model.loss_chance = 0.0
            print("x = 0.0 now - testing performance with train=true test=false")
            im1 = model(im.unsqueeze(0).to(device))
            im1 = im1.squeeze(0).cpu().detach()
    
            # Plot the reconstructed image
            plt.subplot(1, 3, 3)
            plt.imshow(np.transpose(im1/255, (1, 2, 0)))
            plt.title('Reconstructed Image_0.0')
            plt.axis('off')
            plt.show()

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
        for batch_idx, (data, targets) in enumerate(tqdm(train_loader)):
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
    print("x = ", x)
    check_accuracy(train_loader, model, device)
    check_accuracy(val_loader, model, device)
    for x in p:
        model.loss_chance = x
        print("x: ",x)
        check_accuracy(train_loader, model, device)
        check_accuracy(val_loader, model, device)
        print("="*30)
        
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
    
    
