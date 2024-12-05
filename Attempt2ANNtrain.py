import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim




random_seed = 0
torch.manual_seed(random_seed)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


batch_size = 32

train_dataset = datasets.CIFAR10('./data',
                               train=True,
                               download=True,
                               transform=transforms.ToTensor())

validation_dataset = datasets.CIFAR10('./data',
                                    train=False,
                                    transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)#, generator=torch.Generator(device=device))

validation_loader = torch.utils.data.DataLoader(dataset=validation_dataset,
                                                batch_size=batch_size,
                                                shuffle=False)





# class Net(nn.Module):
#     def __init__(self, reg_strength=0.01, clip_value=1.0):
#         super(Net, self).__init__()
#         self.fc1 = nn.Linear(28 * 28, 100, bias=False)
#         self.fc2 = nn.Linear(100, 10, bias=False)
#         self.clip_value=clip_value


#     def forward(self, x):
#             x = x.view(-1, 28*28)
#             x = F.relu(self.fc1(x))
#             x = self.fc2(x)
#             return F.log_softmax(x)
#     def clip_weights(self):
#         # Clip the weights of fc1 and fc2 to be within the range [-clip_value, clip_value]
#         for layer in [self.fc1, self.fc2]:
#             for param in layer.parameters():
#                 param.data = torch.clamp(param.data, -self.clip_value, self.clip_value)
#     def normalize_weights(model):
#         with torch.no_grad():
#             model.fc1.weight.data /= 10.0
#             model.fc2.weight.data /= 10.0

class Net(nn.Module):
    def __init__(self, reg_strength=0.01, clip_value=1.0):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(3*32*32, 512, bias=False)
        # self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10, bias=False)
        self.clip_value=clip_value


    def forward(self, x):
            x = x.view(-1, 3*32*32)
            x = F.relu(self.fc1(x))
            # x = self.dropout(x)
            x = self.fc2(x)
            return F.log_softmax(x)
    def clip_weights(self):
        # Clip the weights of fc1 and fc2 to be within the range [-clip_value, clip_value]
        for layer in [self.fc1, self.fc2]:
            for param in layer.parameters():
                param.data = torch.clamp(param.data, -self.clip_value, self.clip_value)
    # def normalize_weights(model):
    #     with torch.no_grad():
    #         model.fc1.weight.data /= 10.0
    #         model.fc2.weight.data /= 10.0
    #     self.fc1 = nn.Linear(28 * 28, 1000)
    #     self.fc2 = nn.Linear(1000, 10)
    #     self.reg_strength = reg_strength

    # def forward(self, x):
    #     x = x.view(-1, 28*28)
    #     x = F.relu(self.fc1(x))
    #     x = self.fc2(x)

    #     # L2 regularization term
    #     l2_reg = self.reg_strength * (torch.norm(self.fc1.weight) + torch.norm(self.fc2.weight))

    #     return F.log_softmax(x, dim=1) - l2_reg  # Subtract regularization term from output


model = Net().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5, weight_decay=.001)
criterion = nn.CrossEntropyLoss()
# print(type(model.children().next()))


def train(epoch, log_interval=100):
 model.train()
 for batch_idx, (data, target) in enumerate(train_loader):
     data = data.to(device)
     target = target.to(device)
     optimizer.zero_grad()
     output = model(data)
     loss = criterion(output, target)
     loss.backward()
     optimizer.step()
     # model.clip_weights()
     if batch_idx % log_interval == 0:
        print(output[0])
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
                   100. * batch_idx / len(train_loader), loss.data.item()))
 torch.save(model.state_dict(), 'trained_model_cf_512_dropout_seq.pt')
        
def validate(loss_vector, accuracy_vector):
    model.eval()
    val_loss, correct = 0, 0
    for data, target in validation_loader:
        data = data.to(device)
        target = target.to(device)
        output = model(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()
        model.clip_weights()

    val_loss /= len(validation_loader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(validation_loader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(validation_loader.dataset), accuracy))



epochs = 25

lossv, accv = [], []
for epoch in range(1, epochs + 1):
    train(epoch)
    validate(lossv, accv)


