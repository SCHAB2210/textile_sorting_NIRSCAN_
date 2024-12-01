import numpy as np
import pandas as pd
import os
import PIL  # install pillow - pip install Pillow
import PIL.Image
import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import datetime

# Function to read and process data
def read_data(path):
    appended_data = []
    df = [pd.read_csv(filename, header=21) for filename in glob.glob(path)]
    peak_abs = np.zeros((np.shape(df)[0]))
    for file in range(np.shape(df)[0]):
        df[file]['wavelength'] = pd.to_numeric(df[file]['Wavelength (nm)'], errors='coerce')
        df[file]['absorbance'] = pd.to_numeric(df[file]['Absorbance (AU)'], errors='coerce')
        df[file]['absorbance'] = df[file]['absorbance'] / np.max(df[file]['absorbance'])
        df[file] = df[file].drop(
            ['wavelength', 'Absorbance (AU)', 'Reference Signal (unitless)', 
             'Sample Signal (unitless)', 'Wavelength (nm)'], axis=1
        )
        appended_data.append(df[file].T)  # Transpose for row-wise appending
    if appended_data:  # Check if appended_data is not empty
        appended_data = pd.concat(appended_data)  # Combine all files into one DataFrame
    else:
        appended_data = pd.DataFrame()  # Return an empty DataFrame if no data
    return appended_data, peak_abs

# Cotton
path_cotton = r'samples\cotton\**\*.csv'
appended_data_cotton, peak_abs_cotton = read_data(path_cotton)
if not appended_data_cotton.empty:
    class_cotton = np.zeros(len(appended_data_cotton)).astype(int)  # Class 0
    appended_data_cotton.insert(np.shape(appended_data_cotton)[1], "class", class_cotton)
    print(f"Cotton data loaded: {len(appended_data_cotton)} samples.")

# Wool
path_wool = r'samples\wool\**\*.csv'
appended_data_wool, peak_abs_wool = read_data(path_wool)
if not appended_data_wool.empty:
    class_wool = np.ones(len(appended_data_wool)).astype(int)  # Class 1
    appended_data_wool.insert(np.shape(appended_data_wool)[1], "class", class_wool)
    print(f"Wool data loaded: {len(appended_data_wool)} samples.")

# Polyester
path_polyester = r'samples\polyester\**\*.csv'
appended_data_polyester, peak_abs_polyester = read_data(path_polyester)
if not appended_data_polyester.empty:
    class_polyester = np.ones(len(appended_data_polyester)).astype(int) * 2  # Class 2
    appended_data_polyester.insert(np.shape(appended_data_polyester)[1], "class", class_polyester)
    print(f"Polyester data loaded: {len(appended_data_polyester)} samples.")

# Unknown
path_unknown = r'samples\unknown\*.csv'
print(f"Searching for unknown class files in: {path_unknown}")
appended_data_unknown, peak_abs_unknown = read_data(path_unknown)
if not appended_data_unknown.empty:
    class_unknown = np.ones(len(appended_data_unknown)).astype(int) * 3  # Class 3
    appended_data_unknown.insert(np.shape(appended_data_unknown)[1], "class", class_unknown)
    print(f"Unknown data loaded: {len(appended_data_unknown)} samples.")
else:
    print("No data found for the 'unknown' class.")

# Concatenate all data
data = [appended_data_cotton, appended_data_wool, appended_data_polyester, appended_data_unknown]
data = [df for df in data if not df.empty]  # Filter out empty dataframes
if data:
    data = pd.concat(data)
else:
    data = pd.DataFrame()  # Return an empty DataFrame if no data

# Save to CSV
output_path = r'samples\data_cotton_wool_polyester_unknown.csv'
data.to_csv(output_path, index=False)

data = pd.read_csv(r'samples\data_cotton_wool_polyester_unknown.csv')

# Split data into features and labels
y = data['class']
x = data.drop(columns=['class'])

# Stratified train-test split
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Scale inputs and expand dimensions for Conv1D
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encoding for target variables
enc = OneHotEncoder()
y_train_ohe = enc.fit_transform(y_train.values.reshape(-1, 1)).toarray()
y_val_ohe = enc.transform(y_val.values.reshape(-1, 1)).toarray()
y_test_ohe = enc.transform(y_test.values.reshape(-1, 1)).toarray()

# Metrics for the model
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
    tf.keras.metrics.AUC(name='auc', curve='PR'),
]

# Define the model
def make_model(metrics=METRICS):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(8, kernel_size=8, input_shape=(x_train.shape[1], 1), activation='relu'),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(16, 8, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu', activity_regularizer=tf.keras.regularizers.L2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(4, activation='softmax')
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=metrics)
    return model

# Create and train the model
model = make_model()
history = model.fit(
    x_train, y_train_ohe,
    validation_data=(x_val, y_val_ohe),
    epochs=50,
    batch_size=32,
    verbose=1
)

# Evaluate the model
print("\nEvaluating on validation data:")
model.evaluate(x_val, y_val_ohe, batch_size=32)

# Generate predictions
test_predictions_baseline = model.predict(x_test)

# Class names
class_names = ['Cotton', 'Wool', 'Polyester', 'Unknown']

# Plot confusion matrix and classification report
def plot_cm(labels, predictions, class_names):
    predicted_labels = np.argmax(predictions, axis=1)
    cm = confusion_matrix(labels, predicted_labels)
    cmn = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names, ax=ax[0])
    ax[0].set_title('Confusion Matrix')
    sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names, ax=ax[1])
    ax[1].set_title('Normalized Confusion Matrix')
    plt.tight_layout()
    plt.show()
    print('Classification Report:\n', classification_report(labels, predicted_labels, target_names=class_names))

plot_cm(y_test, test_predictions_baseline, class_names)
