import pandas as pd
import numpy as np
import glob
from tensorflow.keras.models import load_model

# Define the preprocessing function
def read_data(path):
    appended_data = []
    df = [pd.read_csv(filename, header=21) for filename in glob.glob(path)]
    for file in range(len(df)):
        df[file]['wavelength'] = pd.to_numeric(df[file]['Wavelength (nm)'], errors='coerce')
        df[file]['absorbance'] = pd.to_numeric(df[file]['Absorbance (AU)'], errors='coerce')
        df[file]['absorbance'] = df[file]['absorbance'] / np.max(df[file]['absorbance'])
        df[file] = df[file].drop(['wavelength', 'Absorbance (AU)', 'Reference Signal (unitless)', 
                                  'Sample Signal (unitless)', 'Wavelength (nm)'], axis=1)
        appended_data.append(df[file].T)
    appended_data = pd.concat(appended_data)
    return appended_data

# Define the prediction function
def predict_label(data_path, model_path):
    data = read_data(data_path)

    # Convert the new data to a NumPy array
    data_arr = np.array(data)

    # Load the trained model
    model = load_model(model_path)

    # Make predictions
    probabilities = model.predict(data_arr)
    if probabilities.ndim == 1:  # Ensure 2D shape for probabilities
        probabilities = np.expand_dims(probabilities, axis=0)
    predicted_label = np.argmax(probabilities, axis=1)[0]

    return predicted_label
