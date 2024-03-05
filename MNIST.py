import os
import pickle

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from nodes_classes import NeuralODE, train, test

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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Specify model to load/create
file_name = 'tmp/models/MNIST.pickle'
os.makedirs(os.path.dirname(file_name), exist_ok=True)
train_losses = []
test_accuracies = []
num_epochs = 2
criterion = nn.CrossEntropyLoss()
model_loaded = False

try:
    with open(file_name, 'rb') as f:
        model = pickle.load(f)

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=True)

    for epoch in range(num_epochs):
        test_accuracy = test(model, test_loader, criterion, device)
        test_accuracies.append(test_accuracy)

    model_loaded = True
except FileNotFoundError:
    # Initialize the model, loss function, and optimizer
    model = NeuralODE().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)

    for epoch in range(num_epochs):
        train_loss = train(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        test_accuracy = test(model, test_loader, criterion, device)
        test_accuracies.append(test_accuracy)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

    # Save Model
    with open(file_name, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting the training loss and test accuracy
fig_accuracy = plt.figure(figsize=(10, 5))

if model_loaded:
    ax_accuracy = fig_accuracy.add_subplot(111)
    ax_accuracy.plot(test_accuracies)
    ax_accuracy.set_xlabel('Trial')
    ax_accuracy.set_ylabel('Test Accuracy (%)')
    ax_accuracy.set_title('Test Accuracies')
else:
    ax_loss = fig_accuracy.add_subplot(121)
    ax_loss.plot(train_losses)
    ax_loss.set_xlabel('Epoch')
    ax_loss.set_ylabel('Training Loss')
    ax_loss.set_title('Training Loss vs. Epoch')

    ax_accuracy = fig_accuracy.add_subplot(122)
    ax_accuracy.plot(test_accuracies)
    ax_accuracy.set_xlabel('Epoch')
    ax_accuracy.set_ylabel('Test Accuracy (%)')
    ax_accuracy.set_title('Test Accuracy vs. Epoch')

fig_accuracy.tight_layout()

plt.show()

