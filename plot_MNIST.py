import pickle
from matplotlib import pyplot as plt


# Load models
with open('canonical/MNIST/Non_NODE_Baseline.pickle', 'rb') as f:
    model_non_node = pickle.load(f)
with open('canonical/MNIST/NODE_5stepEuler.pickle', 'rb') as f:
    model_node_5 = pickle.load(f)
with open('canonical/MNIST/NODE_20stepEuler.pickle', 'rb') as f:
    model_node_20 = pickle.load(f)

models = (model_non_node, model_node_5, model_node_20)
# models = (model_node_5, model_node_20)
# models = (model_node_20,)
model_labels = ('Non-NODE', 'NODE (5 Euler steps)', 'NODE (20 Euler steps)')

# Plots -------------------------------------------------------------------------------------------------------------- #
fontsize = 20
linewidth = 3

fig_losses = plt.figure(figsize=(10, 5))
ax_loss = fig_losses.add_subplot(111)
ax_loss.grid()
for model, model_label in zip(models, model_labels):
    ax_loss.plot(model.train_losses, label=model_label, linewidth=linewidth)
ax_loss.set_xlabel('Epoch', fontsize=fontsize)
ax_loss.set_ylabel('Training Loss', fontsize=fontsize)
ax_loss.set_title('Training Loss vs. Epoch', fontsize=fontsize)
ax_loss.legend(fontsize=fontsize)
fig_losses.tight_layout()
fig_losses.savefig('tmp/MNIST_training_losses.svg')
fig_losses.savefig('tmp/MNIST_training_losses.eps')
fig_losses.savefig('tmp/MNIST_training_losses.png')

fig_accuracy = plt.figure(figsize=(10, 5))
ax_accuracy = fig_accuracy.add_subplot(111)
ax_accuracy.grid()
for model, model_label in zip(models, model_labels):
    ax_accuracy.plot(model.test_accuracies, label=model_label, linewidth=linewidth)
ax_accuracy.set_xlabel('Epoch', fontsize=fontsize)
ax_accuracy.set_ylabel('Test Accuracy (%)', fontsize=fontsize)
ax_accuracy.legend(fontsize=fontsize)
ax_accuracy.set_title('Test Accuracy vs. Epoch', fontsize=fontsize)
fig_accuracy.tight_layout()
fig_accuracy.savefig('tmp/MNIST_training_accuracies.svg')
fig_accuracy.savefig('tmp/MNIST_training_accuracies.eps')
fig_accuracy.savefig('tmp/MNIST_training_accuracies.png')

plt.show()
