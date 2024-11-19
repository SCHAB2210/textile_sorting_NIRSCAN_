import pandas as pd
import numpy as np
import glob
from tensorflow.keras.models import load_model

# Define the preprocessing function
def read_data(path):
    appended_data = []
    df = [pd.read_csv(filename, header=21) for filename in glob.glob(path)]
    peak_abs = np.zeros((np.shape(df)[0]))
    for file in range(np.shape(df)[0]):
        df[file]['wavelength'] = pd.to_numeric(df[file]['Wavelength (nm)'], errors='coerce')
        df[file]['absorbance'] = pd.to_numeric(df[file]['Absorbance (AU)'], errors='coerce')
        df[file]['absorbance'] = df[file]['absorbance'] / np.max(df[file]['absorbance'])
        df[file] = df[file].drop(['wavelength', 'Absorbance (AU)', 'Reference Signal (unitless)', 'Sample Signal (unitless)', 'Wavelength (nm)'], axis=1)
        appended_data.append(df[file].T)
    appended_data = pd.concat(appended_data)
    return appended_data, peak_abs

def predict_label(data_path, model_path):
    data, _ = read_data(data_path)

    # Convert the new data to a NumPy array
    data_arr = np.array(data)

    # Load the trained model
    model = load_model(model_path)

    # Make predictions
    predictions = model.predict(data_arr)

    # Convert predictions to labels (if necessary)
    predicted_label = np.argmax(predictions, axis=1)[0]

    return predicted_label
     
