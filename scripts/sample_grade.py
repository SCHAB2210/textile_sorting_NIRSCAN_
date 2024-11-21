import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("data_cotton_wool_polyester.csv")

y = data['class']
x = data.drop(columns=['class'])
x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20, random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=40)

peak_indices = {}
for label in np.unique(y_train):
    indices = np.argwhere(y_train == label).flatten()
    max_peaks = x_train.iloc[indices].idxmax(axis=1)
    lowest_peak_index = indices[np.argmin(max_peaks)]
    highest_peak_index = indices[np.argmax(max_peaks)]
    mean_sample = x_train.iloc[indices].mean(axis=0)
    
    peak_indices[label] = (lowest_peak_index, highest_peak_index, mean_sample)

# Create subplots
num_classes = len(peak_indices)
fig, axs = plt.subplots(num_classes, 1, figsize=(10, 6 * num_classes), constrained_layout=True)

for i, (label, (lowest_peak_index, highest_peak_index, mean_sample)) in enumerate(peak_indices.items()):
    lowest_peak_value = x_train.iloc[lowest_peak_index].max()
    highest_peak_value = x_train.iloc[highest_peak_index].max()
    mean_highest_peak_value = mean_sample.max()

    lowest_peak_x = x_train.iloc[lowest_peak_index].idxmax()
    highest_peak_x = x_train.iloc[highest_peak_index].idxmax()
    mean_peak_x = mean_sample.idxmax()
    difference = abs(float(highest_peak_x) - float(lowest_peak_x))
    average = (float(highest_peak_x) + float(lowest_peak_x)) / 2

    if difference != 0:
        difference = (difference / average) * 100

    print(f'Class {label}:')
    print(f'  Lowest x = {lowest_peak_x}')
    print(f'  Highest x = {highest_peak_x}')
    print(f'  Mean x = {mean_peak_x}')
    print(f'  Percental Deviation = {difference:.2f} %')

    # Plotting
    x_values = np.arange(len(x_train.iloc[lowest_peak_index])) * 10
    axs[i].plot(x_values, x_train.iloc[lowest_peak_index], label='Lowest Deviation', linestyle='--', color='green', alpha=0.8)
    axs[i].plot(x_values, x_train.iloc[highest_peak_index], label='Highest Deviation', linestyle=':', color='red', alpha=0.8)
    axs[i].plot(x_values, mean_sample, label='Mean Sample', linestyle='-', color='blue', alpha=0.8)
    axs[i].set_title(f'Sample Comparisons for Class {label}')
    axs[i].set_xlabel('Wavelength (10 units per index)')
    axs[i].set_ylabel('Absorbance')
    axs[i].legend()
    axs[i].grid()

plt.show()
