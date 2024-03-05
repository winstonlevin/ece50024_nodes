import pickle

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# Define transformations
# transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize((img_mean,), (img_std,))]
# input tensor values used in Jupyter code: img_std = 0.3081, img_mean = 0.1307, or use 0.5 for both
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
train_set = datasets.MNIST('tmp/data/', download=True, train=True, transform=transform)
test_set = datasets.MNIST('tmp/data/', download=True, train=False, transform=transform)

# Data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
# train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)


# Define the Neural ODE architecture
class ODEFunc(nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # in_channel (1 for grayscale), out_channels
        self.relu = nn.ReLU(inplace=True) # inplace=True argument modifies the input tensor directly, saving memory.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.nfe = 0  # num of function evals

    # forward pass of the neural network
    def forward(self, x):
        self.nfe += 1
        out = self.conv1(x)  # apply first convolution layer to input tensor x
        out = self.relu(out)  # apply ReLU function element-wise to output
        out = self.conv2(out)  # apply second convolution layer to output
        out = self.relu(out)  # apply ReLU function element-wise to output
        return out


class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__()
        self.feature = ODEFunc()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.feature(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def train(_model, _train_loader, _optimizer, _criterion): #training func
    _model.train()
    running_loss = 0.0
    for inputs, labels in _train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        _optimizer.zero_grad()
        outputs = _model(inputs)
        loss = _criterion(outputs, labels)
        loss.backward()
        _optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(_train_loader.dataset)


def test(_model, _test_loader, _criterion):  #testing func
    _model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in _test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = _model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, and optimizer
model = NeuralODE().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 2
train_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    test_accuracy = test(model, test_loader, criterion)
    test_accuracies.append(test_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Save Model
with open('tmp/models/MNIST.data') as file:
    pickle.dump(model, file)

# # Plotting the training loss and test accuracy
# fig_accuracy = plt.figure(figsize=(10, 5))
# ax_loss = fig_accuracy.add_subplot(121)
# ax_loss.plot(train_losses)
# ax_loss.xlabel('Epoch')
# ax_loss.ylabel('Training Loss')
# ax_loss.title('Training Loss vs. Epoch')
#
# ax_accuracy = fig_accuracy.add_subplot(122)
# ax_accuracy.plot(test_accuracies)
# ax_accuracy.xlabel('Epoch')
# ax_accuracy.ylabel('Test Accuracy (%)')
# ax_accuracy.title('Test Accuracy vs. Epoch')
#
# fig_accuracy.tight_layout()
#
# plt.show()

