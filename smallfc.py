import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import os


# Define the fully connected network
class FCModel(nn.Module):
    def __init__(self):
        super(FCModel, self).__init__()
        self.fc1 = nn.Linear(28*28, 100, bias=False)
        self.fc2 = nn.Linear(100, 10, bias=False)
        # self.fc3 = nn.Linear(20, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        # x = self.fc3(x)
        return x

# Define intensity for scaling
intensity = 1.0

# Load MNIST dataset
#I think intensity is not doing anything here, but could if we up it to 128 like snn
transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)])
train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Training the model
model = FCModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()  # Set the model to training mode
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in train_loader:
        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Apply weight bounding/clipping
        # with torch.no_grad():
        #     for param in model.parameters():
        #         param.data.clamp_(min=0)
        
        running_loss += loss.item()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%')
  
# Save the trained model
torch.save(model.state_dict(), 'fc_model.pth')
