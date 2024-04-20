import torch
import pandas as pd
import matplotlib.pyplot as plt
import os

from weather_classes import WeatherDataSet, WeatherPredictor, create_matrix_from_weights, weather_weighted_loss_function
from nodes_classes import EulerIntegrator

# =================== Define Hyperparameters =================== 
batch_size = 64
max_epochs = 50
n_prior_states = 1  # Number of prior states used
n_future_estimates = 4  # Number of future predictions [hr]

# =================== Build Dataset ===================
data_csv_path = './practice3.csv'
dataset_all = WeatherDataSet(
    data_csv_path, n_prior_states=n_prior_states, n_future_estimates=n_future_estimates, normalize=True
)

# Split into 10% Validation / 90% Train
dataset_train, dataset_validation = dataset_all.split_data(frac_validation=0.1)



# df = pd.read_csv('./practice3.csv')  # preprocessed data
# # Extract hour value and convert into a cyclic feature
# df['Hour'] = pd.to_datetime(df['Hour']).dt.hour
# # df['Hour_sin'] = torch.sin(2 * torch.pi * df['Hour'].values / 24.0)  # Make hour data cyclical
# # df['Hour_cos'] = torch.cos(2 * torch.pi * df['Hour'].values / 24.0)
#
# # Select features at time, say t
#
# features = df[['Precip_mm', 'Atm_pressure_mb', 'Global Radiation (Kj/m²)', 'Air_temp_C',  'Rel_Humidity_percent', 'Wind_dir_deg', 'Gust']]
#
# # Drop rows with NaN values resulting from shifting (will essentially remove the last H rows)
# features.dropna(inplace=True)
#
# # Normalize data (default is [0,1])
# features_scaled = scaler_features.fit_transform(features)
# labels_scaled = scaler_labels.fit_transform(labels)
#
# # Split the dataset
# X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels_scaled, test_size=0.2, random_state=seed) #uncomment to visualize scaled dataset, use for training
# # X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=seed) #uncomment to visualize unscaled dataset, but do not use for training
# print("Shape of X_train:", X_train.shape)
# print("Shape of y_train:", y_train.shape)
# print("Shape of X_test:", X_test.shape)
# print("Shape of y_test:", y_test.shape)
#
# # Print first n rows of datasets (default n=5)
# print("X_train:")
# print(pd.DataFrame(X_train).head())
# print("\ny_train:")
# print(pd.DataFrame(y_train).head())
# print("\nX_test:")
# print(pd.DataFrame(X_test).head())
# print("\ny_test:")
# print(pd.DataFrame(y_test).head())
#
# # =================== Build model ===================
# # Multi-Layer Perception NN
# model = Sequential([
#     Input(shape=(features_scaled.shape[1],)),
#     Dense(64, activation='relu'),
#     Dropout(0.1),
#     Dense(64, activation='relu'),
#     Dropout(0.1),
#     Dense(labels_scaled.shape[1])  # Output layer nodes equal to number of labels
# ])
#
# model.summary()
#
# # =================== Compile and train model ===================
# model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
# history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
#
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']
# epochs_range = range(epochs)
#
# # =================== Test model ===================
# predictions = model.predict(X_test)
# # Convert predictions back to original scale
# predictions_original = scaler_labels.inverse_transform(predictions)
#
# # Convert predictions and y_test to DataFrames and visualize
# predictions_df = pd.DataFrame(predictions_original, columns=labels.columns)
# y_test_df = pd.DataFrame(y_test, columns=labels.columns)
# print("Predictions:")
# print(predictions_df.head())
# print("\nActual Values (Labels):")
# print(y_test_df.head())
#
# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print("Test Accuracy:", test_accuracy)
#
# # =================== Plot Training/Validation Loss and Accuracy ===================
# plt.figure(1, figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label='Training Accuracy')
# plt.plot(epochs_range, val_acc, label='Validation Accuracy')
# plt.legend(loc='lower right')
# plt.title(f'Training and Validation Accuracy, H={H}')
#
# plt.figtext(0.5, 0.025, f'Test Accuracy: {test_accuracy:.2f}', ha='center', fontsize=30)
#
# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label='Training Loss')
# plt.plot(epochs_range, val_loss, label='Validation Loss')
# plt.legend(loc='upper right')
# plt.title(f'Training and Validation Loss, H={H}')
#
# if not os.path.exists("./plots"):
#     os.makedirs("./plots")
#
# plt.savefig(f"./plots/loss_H={H}.jpg")
#
# plt.show()
#
