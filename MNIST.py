import os
import pickle
import time

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from nodes_classes import MNISTClassifier, train, test, solve_ivp_euler

torch.autograd.set_detect_anomaly(True)  # TODO - Remove

# Hyperparameters
batch_size = 64
n_features = 64

# Load MNIST dataset and create loaders for training/testing
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_set = datasets.MNIST('tmp/data/', download=True, train=True, transform=transform)
test_set = datasets.MNIST('tmp/data/', download=True, train=False, transform=transform)

train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False)
small_data_set = False

if small_data_set:
    # To simply ensure implementation works, use small subset of data which is fast to train/test
    n_data = 21 * batch_size
    train_loader.dataset.data = train_loader.dataset.data[0:n_data, :, :]
    train_loader.dataset.targets = train_loader.dataset.targets[0:n_data]
    test_loader.dataset.data = test_loader.dataset.data[0:n_data, :, :]
    test_loader.dataset.targets = test_loader.dataset.targets[0:n_data]


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_epochs = 10
criterion = nn.CrossEntropyLoss()


# Initialize the model, loss function, and optimizer
def integrator(fun, t_span, y0):
    return solve_ivp_euler(fun, t_span, y0, n_steps=20)


model = MNISTClassifier(n_features=n_features, use_node=True, integrator=integrator).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device, verbose=True)
    model.train_losses.append(train_loss)
    test_accuracy = test(model, test_loader, device)
    model.test_accuracies.append(test_accuracy)
    print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Save Model
current_time = time.gmtime()
date = f'{current_time.tm_year:04d}-{current_time.tm_mon:02d}-{current_time.tm_mday:02d}'
hour = f'{current_time.tm_hour:02d}-{current_time.tm_min:02d}-{current_time.tm_sec:02d}'
file_name = f'tmp/models/MNIST_{date}_{hour}.pickle'
os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Make directory if it does not yet exist
with open(file_name, 'wb') as f:
    pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting the training loss and test accuracy
fig_accuracy = plt.figure(figsize=(10, 5))

ax_loss = fig_accuracy.add_subplot(121)
ax_loss.plot(model.train_losses)
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Training Loss')
ax_loss.set_title('Training Loss vs. Epoch')

ax_accuracy = fig_accuracy.add_subplot(122)
ax_accuracy.plot(model.test_accuracies)
ax_accuracy.set_xlabel('Epoch')
ax_accuracy.set_ylabel('Test Accuracy (%)')
ax_accuracy.set_title('Test Accuracy vs. Epoch')

fig_accuracy.tight_layout()

plt.show()
