import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchdiffeq

# Original Jupyter code demo https://nbviewer.org/github/msurtsukov/neural-ode/blob/master/Neural%20ODEs.ipynb

# ===================== Dataset Preprocessing ==========================
# Define transformations to be done to input images
# 0.5, 0.5 are parameters used to normalize the data, assuming grayscale image, like MNIST 
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]) #values from code demo: img_std = 0.3081, img_mean = 0.1307, idk why
# Load MNIST dataset (not sure what the )
train_set = datasets.MNIST('data/', download=True, train=True, transform=transform)
test_set = datasets.MNIST('data/', download=True, train=False, transform=transform)
# Data loaders
train_loader = torch.utils.data.DataLoader(train_set, batch_size = 64, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size = 64, shuffle=False)


# ===================== Defining the NN Model ==========================
# Define the ODEFunc class to represent a neural network
# (based on init of ContinuousNeuralMNISTClassifier())
class ODEFunc(nn.Module): #superclass: nn.Module
    def __init__(self):
        super(ODEFunc, self).__init__() #initialize superclass 
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1) # in_channel (1 channel  per pixel for grayscale [0,255]), out_channels (64 chosen to reflect code demo)
        self.relu = nn.ReLU(inplace=True) # inplace=True argument allows the relu function to modify the input tensor directly, saving memory.
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.nfe = 0  # num of function evals

    # forward pass of the neural network
    def forward(self, x):
        self.nfe += 1
        out = self.conv1(x) # apply first convolution layer to input tensor x
        out = self.relu(out) # apply ReLU activation function element-wise to output
        out = self.conv2(out) # apply second convolution layer to output
        out = self.relu(out) # apply ReLU activation function element-wise to output
        return out

# Define the NeuralODE class to represent the entire architecture
# (based on ContinuousNeuralMNISTClassifier())
class NeuralODE(nn.Module):
    def __init__(self):
        super(NeuralODE, self).__init__() 
        self.feature = ODEFunc()
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.feature(x) # pass dataset into NN
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ===================== Defining Functions to train and test the model ==========================
# Training Loop for the model
def train(model, train_loader, optimizer, criterion): #training func
    model.train() #set model to training mode
    running_loss = 0.0
    for inputs, labels in train_loader: #iterate through training data via pytorch data loader, batch of input data w/ corr. labels
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) #performs forward pass through NN model
        loss = criterion(outputs, labels) # calculates loss using loss function (Cross Entropy Loss)
        loss.backward() #backpropogation
        optimizer.step() #adjust parameters of model to minimize loss function
    #     running_loss += loss.item() * inputs.size(0) # add loss 
    # return running_loss / len(train_loader.dataset)
        running_loss += loss.item() * inputs.size(0) # loss x batch size
    return running_loss / len(train_loader.dataset) # total loss / num_samples, average loss for this epoch of training

def test(model, test_loader, criterion): #testing func
    model.eval() #test mode
    correct = 0
    total = 0
    with torch.no_grad(): #Tell pytorch to not track gradients, reducing computational speeds
        for inputs, labels in test_loader: #iterate through test data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1) # predicted class index (the index of the maximum value within the output tensor)
            total += labels.size(0)
            correct += (predicted == labels).sum().item() #calculate the number of correct predictions and convert to python scalar
    accuracy = 100 * correct / total
    return accuracy


# Device configuration, utilize GPUs compatible with cuda to speed up training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model, loss function, and optimizer
model = NeuralODE().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Execute the training and testing loops
num_epochs = 10
train_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, optimizer, criterion)
    train_losses.append(train_loss)
    test_accuracy = test(model, test_loader, criterion)
    test_accuracies.append(test_accuracy)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%")

# Plotting the training loss and test accuracy
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Epoch')

plt.subplot(1, 2, 2)
plt.plot(test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Test Accuracy (%)')
plt.title('Test Accuracy vs. Epoch')

plt.tight_layout()
plt.show()

