import os
import pickle
import time

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from weather_classes import WeatherDataSet, WeatherPredictor, create_matrix_from_weights, \
    weather_weighted_loss_function, train, test, test_state
from nodes_classes import RKIntegrator, EulerIntegrator

torch.autograd.set_detect_anomaly(True)

# =================== Define Hyperparameters =================== 
batch_size = 64
max_epochs = 50
max_epochs_without_improvement = 2
n_prior_states = 1  # Number of prior states used
n_future_estimates = 8  # Number of future predictions [hr]

# States are: precipitation, pressure, radiation, temperature, relative humidity, wind direction, and wind speed
weight_high = 1.
weight_medium = 1e-1
weight_low = 1e-3
# weights = torch.tensor((weight_medium, weight_medium, weight_low, weight_high, weight_medium, weight_low, weight_low))
weights = torch.tensor((weight_low, weight_low, weight_low, weight_high, weight_low, weight_low, weight_low))

discount_factor = 0.9

learning_rate = 1e-3

use_node = False
integrator = EulerIntegrator(n_steps=20)
# integrator = RKIntegrator(min_time_step=1./20.)

# =================== Build Dataset ===================
data_csv_path = './tmp/data/Weather/north_processed.csv'
preprocess_path = './tmp/data/WeatherDataset/time_state.csv'  # Path to save Numpy Array of normalized state

dataset_all = WeatherDataSet(
    csv_path=data_csv_path, preprocess_path=preprocess_path,
    n_prior_states=n_prior_states, n_future_estimates=n_future_estimates, normalize=True, dtype=torch.float32
)
state_names = dataset_all.columns_to_use.copy()
idx_time_state_temperature = state_names.index('Air_temp_C')
state_names.remove('Hour')
idx_state_temperature = state_names.index('Air_temp_C')
scale_temperature = dataset_all.half_range[idx_time_state_temperature]

# Split into 10% Validation / 90% Train
dataset_train, dataset_validation = dataset_all.split_data(frac_validation=0.1)
n_states = dataset_train.n_states

train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
validation_loader = torch.utils.data.DataLoader(dataset_validation, batch_size=batch_size, shuffle=False)

# =================== Compile and train model ===================
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

prediction_weights = create_matrix_from_weights(
    weights, num_future_predictions=n_future_estimates, discount_for_future=discount_factor, return_matrix=False
)


def criterion(_output, _target):
    return weather_weighted_loss_function(_output, _target, prediction_weights, reduction='mean')


model = WeatherPredictor(
    n_states=n_states, n_prior_states=n_prior_states, n_predictions=n_future_estimates, n_layers=3,
    activation_type='SiLU', use_node=use_node, integrator=integrator, dtype=torch.float32
).to(device)
model.weights = prediction_weights
for idx_predict in range(n_future_estimates):
    # Create one list for each prediction horizon
    model.temperature_accuracies.append([])

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

epochs_without_improvement = 0
test_loss_prev = torch.inf
for epoch in range(max_epochs):
    train_loss = train(model, train_loader, optimizer, criterion, device, verbose=True)
    model.train_losses.append(train_loss)
    test_loss = test(model, validation_loader, criterion, device)
    model.test_losses.append(test_loss)
    temp_accuracy = scale_temperature * test_state(model, validation_loader, device, idx_state=idx_state_temperature)

    for idx_predict, temp_acc in enumerate(temp_accuracy):
        model.temperature_accuracies[idx_predict].append(temp_acc.item())

    print(f"[Epoch {epoch + 1}/{max_epochs}] "
          f"Train Loss: {train_loss}, Validation Loss: {test_loss}, 1-hour Temp. Err.: {temp_accuracy[0]}")

    # Detect overtraining
    if test_loss < test_loss_prev:
        epochs_without_improvement = 0  # Reset
    else:
        epochs_without_improvement += 1

    if epochs_without_improvement > max_epochs_without_improvement:
        print('Overtraining detected! Ending training.')
        break
    test_loss_prev = test_loss

    # Save Model
    current_time = time.gmtime()
    date = f'{current_time.tm_year:04d}-{current_time.tm_mon:02d}-{current_time.tm_mday:02d}'
    hour = f'{current_time.tm_hour:02d}-{current_time.tm_min:02d}-{current_time.tm_sec:02d}'
    file_name = f'tmp/models/Weather_{date}_{hour}.pickle'
    os.makedirs(os.path.dirname(file_name), exist_ok=True)  # Make directory if it does not yet exist
    with open(file_name, 'wb') as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

# Plotting the training loss and test accuracy
fig_accuracy = plt.figure(figsize=(10, 5))

ax_loss = fig_accuracy.add_subplot(121)
ax_loss.grid()
ax_loss.plot(model.train_losses, label='Training Set')
ax_loss.plot(model.test_losses, label='Validation Set')
ax_loss.set_xlabel('Epoch')
ax_loss.set_ylabel('Loss')
ax_loss.legend()

delta_str = r"$\Delta$"
ax_acc = fig_accuracy.add_subplot(122)
ax_acc.grid()
for idx, temp_acc in enumerate(model.temperature_accuracies, start=1):
    ax_acc.plot(temp_acc, label=f'{delta_str}t = {idx} hr')
ax_acc.legend()

fig_accuracy.tight_layout()

plt.show()
