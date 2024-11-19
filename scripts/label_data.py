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

def predict_label(path_new_data):
    new_data, _ = read_data(path_new_data)

    # Convert the new data to a NumPy array
    new_data_arr = np.array(new_data)

    # Load the trained model
    model_save_path = r'20241119-085834.h5'
    model = load_model(model_save_path)

    # Make predictions
    predictions = model.predict(new_data_arr)

    # Convert predictions to labels (if necessary)
    predicted_labels = np.argmax(predictions, axis=1)

    return predicted_labels
     
