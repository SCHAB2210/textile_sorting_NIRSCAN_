import numpy as np
import pandas as pd
import os
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Function to process individual sample files
def read_sample(file_path, label):
    """
    Reads a sample CSV file, extracts the 'Absorbance (AU)' data,
    and assigns the given label.
    """
    # Skip metadata rows and read relevant data
    df = pd.read_csv(file_path, skiprows=21)
    
    # Extract 'Absorbance (AU)' column as features
    features = df['Absorbance (AU)'].values
    
    # Create a DataFrame with features and label
    data = pd.DataFrame([features])
    data['class'] = label  # Add the class label
    
    return data

# Process all samples
def process_samples(data_dir, class_labels):
    """
    Reads all sample files in the given directory and assigns class labels.
    """
    all_data = []
    
    for label, folder in class_labels.items():
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                file_path = os.path.join(folder_path, file)
                sample_data = read_sample(file_path, label)
                all_data.append(sample_data)
    
    # Combine all samples into a single DataFrame
    return pd.concat(all_data, axis=0, ignore_index=True)

# Define class labels and corresponding folder names
class_labels = {
    0: "cotton",
    1: "wool",
    2: "polyester",
    3: "unknown",
}

# Directory containing the sample data
data_dir = "samples"  # Adjust to your actual directory path

# Process all samples
data = process_samples(data_dir, class_labels)

# Split features and labels
y = data['class']
x = data.drop(columns=['class'])

# Split into train, validation, and test sets
x_train, x_temp, y_train, y_temp = train_test_split(
    x, y, test_size=0.3, stratify=y, random_state=42
)
x_val, x_test, y_val, y_test = train_test_split(
    x_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
)

# Standardize inputs
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# Expand dimensions for Conv1D
x_train = np.expand_dims(x_train, axis=-1)
x_val = np.expand_dims(x_val, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

# One-hot encoding for target variables
encoder = OneHotEncoder(sparse=False)
y_train_ohe = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_val_ohe = encoder.transform(y_val.values.reshape(-1, 1))
y_test_ohe = encoder.transform(y_test.values.reshape(-1, 1))

# Define model metrics
METRICS = [
    tf.keras.metrics.CategoricalAccuracy(name="accuracy"),
    tf.keras.metrics.AUC(name="auc", curve="PR"),
]

# Define the model
def make_model(metrics=METRICS):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(8, kernel_size=8, input_shape=(x_train.shape[1], 1), activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Conv1D(16, kernel_size=8, padding="same", activation="relu"),
        tf.keras.layers.MaxPooling1D(pool_size=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation="relu", activity_regularizer=tf.keras.regularizers.L2(0.01)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(len(encoder.categories_[0]), activation="softmax"),
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
test_predictions = model.predict(x_test)

# Class names
class_names = encoder.categories_[0]

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
    print("Classification Report:\n", classification_report(labels, predicted_labels, target_names=class_names))

plot_cm(y_test, test_predictions, class_names)
