import torch
import torch.nn as nn


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


def train(_model, _train_loader, _optimizer, _criterion, _device): #training func
    _model.train()
    running_loss = 0.0
    for inputs, labels in _train_loader:
        inputs, labels = inputs.to(_device), labels.to(_device)
        _optimizer.zero_grad()
        outputs = _model(inputs)
        loss = _criterion(outputs, labels)
        loss.backward()
        _optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    return running_loss / len(_train_loader.dataset)


def test(_model, _test_loader, _criterion, _device):  #testing func
    _model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in _test_loader:
            inputs, labels = inputs.to(_device), labels.to(_device)
            outputs = _model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy
