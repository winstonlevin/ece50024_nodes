import pickle
from matplotlib import pyplot as plt
import torch


# Load models
with open('canonical/Weather/Non_NODE_Baseline.pickle', 'rb') as f:
    model_non_node = pickle.load(f)
with open('canonical/Weather/NODE_20stepEuler.pickle', 'rb') as f:
    model_node_20 = pickle.load(f)

models = (model_non_node, model_node_20)
model_labels = ('Non-NODE', 'NODE (20 Euler steps)')

# Plots -------------------------------------------------------------------------------------------------------------- #
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

fontsize = 20
linewidth = 3

fig_losses = plt.figure(figsize=(10, 5))
ax_loss = fig_losses.add_subplot(111)
ax_loss.grid()
loss_plot_max = 0.
loss_plot_min = torch.inf
for idx, (model, model_label) in enumerate(zip(models, model_labels)):
    _train_loss = torch.as_tensor(model.train_losses)
    _test_loss = torch.as_tensor(model.test_losses)
    _loss_plot_max_model = torch.as_tensor(model.test_losses).max()
    _loss_plot_min_model = _train_loss.min().minimum(_test_loss.min()) * 0.99
    if loss_plot_max < _loss_plot_max_model:
        loss_plot_max = _loss_plot_max_model
    if loss_plot_min > _loss_plot_min_model:
        loss_plot_min = _loss_plot_min_model

    ax_loss.plot(
        model.train_losses, linestyle='--', label=model_label + ' (Training)', linewidth=linewidth, color=colors[idx]
    )
    ax_loss.plot(model.test_losses, label=model_label + ' (Validation)', linewidth=linewidth, color=colors[idx])
ax_loss.set_xlabel('Epoch', fontsize=fontsize)
ax_loss.set_ylabel('Loss', fontsize=fontsize)
# ax_loss.set_title('Training Loss vs. Epoch', fontsize=fontsize)
ax_loss.legend(fontsize=fontsize)
ax_loss.set_ylim(loss_plot_min, loss_plot_max)
fig_losses.tight_layout()
fig_losses.savefig('tmp/Weather_losses.svg')
fig_losses.savefig('tmp/Weather_losses.eps')
fig_losses.savefig('tmp/Weather_losses.png')

fig_accuracy = plt.figure(figsize=(10, 5))
ax_accuracy = fig_accuracy.add_subplot(111)
ax_accuracy.grid()
for idx, (model, model_label) in enumerate(zip(models, model_labels)):
    ax_accuracy.plot(
        model.temperature_accuracies[0], linestyle='--',
        label=model_label + ' (1 Hour)', linewidth=linewidth, color=colors[idx]
    )
    ax_accuracy.plot(
        model.temperature_accuracies[-1], linestyle='-',
        label=model_label + f' ({len(model.temperature_accuracies)} Hour)', linewidth=linewidth, color=colors[idx]
    )
ax_accuracy.set_xlabel('Epoch', fontsize=fontsize)
ax_accuracy.set_ylabel(r'Mean Temperature Error [$^\circ$C]', fontsize=fontsize)
ax_accuracy.legend(fontsize=fontsize)
# ax_accuracy.set_title('Test Accuracy vs. Epoch', fontsize=fontsize)
fig_accuracy.tight_layout()
fig_accuracy.savefig('tmp/Weather_temperature_accuracies.svg')
fig_accuracy.savefig('tmp/Weather_temperature_accuracies.eps')
fig_accuracy.savefig('tmp/Weather_temperature_accuracies.png')

plt.show()
