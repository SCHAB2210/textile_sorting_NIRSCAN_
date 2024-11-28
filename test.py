import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import glob
import datetime
import matplotlib.pyplot as plt

# Function to read and process data
def read_data(path):
    appended_data = []
    df = [pd.read_csv(filename, header=21) for filename in glob.glob(path)]
    peak_abs = np.zeros(len(df))  # Placeholder, not currently used
    for file in range(len(df)):
        df[file]['wavelength'] = pd.to_numeric(df[file]['Wavelength (nm)'], errors='coerce')
        df[file]['absorbance'] = pd.to_numeric(df[file]['Absorbance (AU)'], errors='coerce')
        df[file]['absorbance'] = df[file]['absorbance'] / np.max(df[file]['absorbance'])
        df[file] = df[file].drop(['wavelength', 'Absorbance (AU)', 'Reference Signal (unitless)',
                                  'Sample Signal (unitless)', 'Wavelength (nm)'], axis=1)
        appended_data.append(df[file].T)  # Transpose for row-wise appending
    if appended_data:
        appended_data = pd.concat(appended_data, ignore_index=True)  # Combine all files into one DataFrame
    else:
        appended_data = pd.DataFrame()  # Return an empty DataFrame if no data
    return appended_data, peak_abs

# Define paths
base_path = r'samples'
paths = {
    "cotton": os.path.join(base_path, "cotton", "**", "*.csv"),
    "wool": os.path.join(base_path, "wool", "**", "*.csv"),
    "polyester": os.path.join(base_path, "polyester", "**", "*.csv"),
    "unknown": os.path.join(base_path, "unknown", "*.csv"),
}

# Process data for each material
dataframes = []
for material, path in paths.items():
    print(f"Processing {material} data from: {path}")
    appended_data, _ = read_data(path)
    if not appended_data.empty:
        material_class = len(dataframes)  # Assign class based on order (0, 1, 2, ...)
        material_labels = np.full(len(appended_data), material_class, dtype=int)
        appended_data.insert(appended_data.shape[1], "class", material_labels)
        dataframes.append(appended_data)
        print(f"{material.capitalize()} data loaded: {len(appended_data)} samples.")
    else:
        print(f"No data found for {material}.")

# Concatenate all data
if dataframes:
    data = pd.concat(dataframes, ignore_index=True)
else:
    data = pd.DataFrame()
    print("No data found for any material. Exiting...")
    exit()

# Save to CSV
output_path = os.path.join(base_path, "data_cotton_wool_polyester_unknown.csv")
data.to_csv(output_path, index=False)
print(f"Data saved to {output_path}")

# Load processed data
data = pd.read_csv(output_path)
y = data['class']
x = data.drop(columns=['class'])

# Split data into training, validation, and test sets
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=40)

# Normalize features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# One-hot encode target variables
enc = OneHotEncoder()
enc.fit(y_train.values.reshape(-1, 1))  # Fit once
y_train_ohe = enc.transform(y_train.values.reshape(-1, 1)).toarray()
y_val_ohe = enc.transform(y_val.values.reshape(-1, 1)).toarray()
y_test_ohe = enc.transform(y_test.values.reshape(-1, 1)).toarray()

# Define metrics
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc'),
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
]

# Build model
def make_model(metrics=METRICS):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(x_train.shape[1],)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')  # 3 classes: cotton, wool, polyester
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=metrics
    )
    return model

# Instantiate and summarize model
model = make_model()
model.summary()

# Define directories and callbacks
log_path = os.path.join(base_path, "logs")
os.makedirs(log_path, exist_ok=True)

# Save model with a timestamp
model_save_path = os.path.join(log_path, f"model_{datetime.datetime.now().strftime('%Y%m%d-%H%M%S')}.h5")

# TensorBoard callback
tensorboard_callback = tf.keras.callbacks.TensorBoard(
    log_dir=log_path,
    histogram_freq=1,
    write_graph=False,
    write_images=False,
    update_freq='epoch',
    profile_batch=0,
    embeddings_freq=0
)

# Early stopping callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# Train model
history = model.fit(
    x_train,
    y_train_ohe,
    validation_data=(x_val, y_val_ohe),
    epochs=50,
    batch_size=32,
    verbose=1,
    callbacks=[tensorboard_callback, early_stopping]
)

# Save the trained model
model.save(model_save_path)
print(f"Model saved at: {model_save_path}")

# Plot training metrics
def plot_metrics(history):
    metrics = ['loss', 'accuracy', 'precision', 'recall']
    plt.figure(figsize=(12, 8))
    for n, metric in enumerate(metrics):
        plt.subplot(2, 2, n + 1)
        plt.plot(history.history[metric], label='Train')
        plt.plot(history.history[f'val_{metric}'], linestyle="--", label='Val')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.show()

plot_metrics(history)
