# %%

import numpy as np
import pandas as pd
import os
import PIL # install pillow - pip install Pillow
import PIL.Image
import tensorflow as tf
#from tensorflow import keras
import tensorflow_datasets as tfds # need to install this seperately - pip install tensorflow_datasets
import pathlib
from skimage import io
import datetime
import matplotlib.pyplot as plt
import h5py # pip install h5py
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.gridspec as gridspec
import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
import glob
import os
import shutil

# %% [markdown]
# # copy .csv files from one location to other
# # use only once when you need to add new class

# %%
# copy .csv files from one location to other
#use only once when you need to add new class


# src_folder = r"C:\Users\ajitj\OneDrive - Universitetet i Agder\PhD_Research\Paper_Writing\Textile Sorting\Data_textile_NIRSCAN\ML_textile_classification_timeseries\Data_NIRSCAN_CSV\polyester-000011\polyester-000011"
# dst_folder = r"C:\Users\ajitj\OneDrive - Universitetet i Agder\PhD_Research\Paper_Writing\Textile Sorting\Data_textile_NIRSCAN\ML_textile_classification_timeseries\data_ml\polyester"

# # Search files with .txt extension in source directory
# pattern = "\*.csv"
# files = glob.glob(src_folder + pattern)

# # move the files with txt extension
# for file in files:
#     # extract file name form file path
#     file_name = os.path.basename(file)
#     shutil.move(file, dst_folder + file_name)
#     #print('Moved:', file)

# %% [markdown]
# # Use this to generate .csv file for ml. Use only once to generate data_cotton_wool_polyester.csv

# %%
#run this cell to generate a .csv file to be used for ml

def read_data(path):
    appended_data = []
    df = [pd.read_csv(filename,header=21) for filename in glob.glob(path)] 
    peak_abs=np.zeros((np.shape(df)[0]))
    for file in range(np.shape(df)[0]):
        #df[file][['wavelength','absorbance', 'reference', 'sample_signal']]=df[file]['data'].str.split(expand=True)
        df[file]['wavelength'] = pd.to_numeric(df[file]['Wavelength (nm)'], errors='coerce')
        df[file]['absorbance'] = pd.to_numeric(df[file]['Absorbance (AU)'], errors='coerce')
        df[file]['absorbance']=df[file]['absorbance']/np.max(df[file]['absorbance'])
         # remove data col, as not needed
        df[file]=df[file].drop(['wavelength', 'Absorbance (AU)', 'Reference Signal (unitless)', 'Sample Signal (unitless)', 'Wavelength (nm)'], axis=1)
        #appended_data.append(df[file]) #appends all values in col
        appended_data.append(df[file].T) #228x1 appends each acq in row -- 900 x 228x1
        
        #peak_abs[file]=df[file]['wavelength'][np.argmax(df[file]['absorbance'])]
    appended_data = pd.concat(appended_data) # all 900 acquisition appended together
    #appended_data_coton.to_excel('appended.xlsx') # write to csv    
    return appended_data, peak_abs


# def read_data(path):
#     appended_data=[]
#     for file in range(np.shape(df)[0]):
#         df[file] = pd.DataFrame(df[file])
#         #df_cotton[file]=df_cotton[file].drop(['wavelength'], axis=1)
#         appended_data.append(df[file]) #228x1
#     appended_data = pd.concat(appended_data) # all 900 acquisition appended together
#     appended_data.to_csv('df_cotton.csv', index=False)

#cotton
path_cotton=r'C:\Users\devTe\Desktop\ML\samples\Cotton\**\*.csv' # path of file to read
appended_data_cotton, peak_abs_cotton = read_data(path_cotton) 
class_cotton=np.zeros(len(appended_data_cotton)).astype(int)
appended_data_cotton.insert(np.shape(appended_data_cotton)[1], "class", class_cotton)

#wool
path_wool=r'C:\Users\devTe\Desktop\ML\samples\wool\**\*.csv'
appended_data_wool, peak_abs_wool =read_data(path_wool) # read file 1000x227x4
class_wool=np.ones(len(appended_data_wool)).astype(int)
appended_data_wool.insert(np.shape(appended_data_wool)[1], "class", class_wool)

#polyester
path_polyester=r'C:\Users\devTe\Desktop\ML\samples\polyester\**\*.csv'
appended_data_polyester, peak_abs_polyester =read_data(path_polyester) # read file 1000x227x4
class_polyester=np.ones(len(appended_data_polyester)).astype(int) * 2
appended_data_polyester.insert(np.shape(appended_data_polyester)[1], "class", class_polyester)

#check
print("Cotton data : \n {}".format (appended_data_cotton.head()))
print("Wool data : \n {}".format (appended_data_wool.head()))
print("Polyester data : \n {}".format (appended_data_polyester.head()))

#concatenate data frames
data = [appended_data_cotton, appended_data_wool, appended_data_polyester]
data=pd.concat(data) # contains both cottorn and wool with class 0 and 1

print("data-head : \n {}".format (data.head()))
print("data-tail : \n {}".format (data.tail()))

#save to csv
data.to_csv('data_cotton_wool_polyester.csv', index=False)  # only use when you want to add new class

#check
data['class'].value_counts()



# %% [markdown]
# # Use this code to load data_cotton_wool_polyester.csv

# %%
data = pd.read_csv('data_cotton_wool_polyester.csv')
#read cotton file, to extract wavelength value. This is same for all acq
path_c=r'C:\Users\devTe\Desktop\ML\samples\Cotton\**\*.csv' # path of file to read
files = glob.glob(path_c)
data_c = pd.concat([pd.read_csv(file, header=21) for file in files], ignore_index=True)
wavelength = data_c['Wavelength (nm)']
wavelength = np.around(wavelength) #truncate to 1 dec place
wavelength = wavelength.to_numpy()


# %%
wavelength[-1]

# %%

cotton_portion=data[0:int(np.shape(data)[0]/3)] #900x229; 900 data samples (acq), 0:227 data, 228 label data frame; data --1800x229
wool_portion=data[int(np.shape(data)[0]/3):2*int(np.shape(data)[0]/3)]
polyester_portion=data[2*int(np.shape(data)[0]/3)::]

#normalize spectral data
cp=cotton_portion.iloc[0][0:228]
wp=wool_portion.iloc[0][0:228]
pp=polyester_portion.iloc[0][0:228]

cp_n=(cp-np.min(cp))/(np.max(cp)-np.min(cp))
wp_n=(wp-np.min(wp))/(np.max(wp)-np.min(wp))
pp_n=(pp-np.min(pp))/(np.max(pp)-np.min(pp))

plt.figure(1)
#plt.title('Data')
plt.plot(wavelength[:228], cp_n, 'g', label='cotton') #cotton_portion.iloc[0][0:228]
plt.plot(wavelength[:228], wp_n, 'b', label='wool')
plt.plot(wavelength[:228], pp_n, 'r', label='polyester')
plt.xlabel('Wavelength (nm)')
plt.ylabel('Amplitude (a.u)')
# Change x-axis tick spacing
plt.xticks(np.arange(wavelength[0], wavelength[227], step=80))  # ticks at 0, 2, 4, ..., 10
plt.legend()
plt.show()


# %%
cp=cotton_portion.iloc[0][0:228]
cp_n=(cp-np.mean(cp))/np.std(cp)
cp_n

# %%
# def read_data(path):
#     appended_data = []
#     df = [pd.read_csv(filename,header=21) for filename in glob.glob(path)] 
#     peak_abs=np.zeros((np.shape(df)[0]))
#     for file in range(np.shape(df)[0]):
#         #df[file][['wavelength','absorbance', 'reference', 'sample_signal']]=df[file]['data'].str.split(expand=True)
#         df[file]['wavelength'] = pd.to_numeric(df[file]['Wavelength (nm)'], errors='coerce')
#         df[file]['absorbance'] = pd.to_numeric(df[file]['Absorbance (AU)'], errors='coerce')
#         df[file]['absorbance']=df[file]['absorbance']/np.max(df[file]['absorbance'])
#          # remove data col, as not needed
#         df[file]=df[file].drop(['Absorbance (AU)', 'Reference Signal (unitless)', 'Sample Signal (unitless)', 'Wavelength (nm)'], axis=1)
#         appended_data.append(df[file])
#         peak_abs[file]=df[file]['wavelength'][np.argmax(df[file]['absorbance'])]
#     appended_data = pd.concat(appended_data) # all 900 acquisition appended together
#     #appended_data_coton.to_excel('appended.xlsx') # write to csv    
#     return df, appended_data, peak_abs



# def read_data(path):
#     append_files=[]
#     for file in range(np.shape(df_cotton)[0]):
#         df_cotton[file] = pd.DataFrame(df_cotton[file])
#         #df_cotton[file]=df_cotton[file].drop(['wavelength'], axis=1)
#         append_files_cotton.append(df_cotton[file].T) #228x1
#     append_files_cotton = pd.concat(append_files_cotton) # all 900 acquisition appended together
#     append_files_cotton.to_csv('df_cotton.csv', index=False)




# path_cotton=r'C:\Users\ajitj\OneDrive - Universitetet i Agder\PhD_Research\Paper_Writing\Textile Sorting\Data_textile_NIRSCAN\ML_textile_classification_timeseries\train\cotton\*.csv' # path of file to read
# df_cotton, appended_data_cotton, peak_abs_cotton =read_data(path_cotton) # read file 1000x227x4
# class_cotton=np.zeros(len(appended_data_cotton)).astype(int)
# appended_data_cotton.insert(2, "class", class_cotton)

# %%
# np.shape(df_cotton), np.shape(appended_data_cotton), 

# append_files_cotton=[]
# #convert df_cotton from list to pd dataframe, save it as csv
# for file in range(np.shape(df_cotton)[0]):
#     df_cotton[file] = pd.DataFrame(df_cotton[file])
#     #df_cotton[file]=df_cotton[file].drop(['wavelength'], axis=1)
#     append_files_cotton.append(df_cotton[file].T) #228x1
# append_files_cotton = pd.concat(append_files_cotton) # all 900 acquisition appended together
# append_files_cotton.to_csv('df_cotton.csv', index=False)

# %% [markdown]
#  # Use this to read csv files and do ml. Use this ONLY when the data is  saved as .csv

# %%
#use this to read csv files and onwards
data = pd.read_csv(r'C:\Users\devTe\Desktop\ML\samples\data_cotton_wool_polyester.csv')
data.head()

# %%
# train test data set

# train_df, val_df = train_test_split(data, test_size=0.20)
# train_df, test_df = train_test_split(train_df, test_size=0.1)

# #training, val and test data size
# print ("Train shape: {} \n Val shape: {} \n Test shape: {} \n" .format(np.shape(train_df), np.shape(val_df), np.shape(test_df))) 
# #np.array( [np.shape(train_df)[0], np.shape(val_df)[0], np.shape(val_df)[0] ]) /np.array([np.shape(data)[0]])
# # train, val, test - 70, 20, 10

# #save to csv
# #data.to_csv('data_cotton_wool.csv', index=False)  

# #check
# data['class'].value_counts()

# %%
# train, test split 

y = data['class']
x = data.drop(columns=['class'])
#x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.2, random_state=42)

x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=0.20,random_state=42)
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.1, random_state=40)


print("Train: \n {}\n".format(y_train.value_counts()))
print("Val: \n {} \n".format(y_val.value_counts()))
print("Test: \n {} \n".format(y_test.value_counts()))

print("Train: \n {} {}\n".format(np.shape(x_train), np.shape(y_train)))
print("Val: \n {} {}\n".format(np.shape(x_val), np.shape(y_val)))
print("Test: \n {} {}\n".format(np.shape(x_test), np.shape(y_test)))

# Train: (1440, 228) (1440,)

#Val: (360, 228) (360,)


# %%
#check few samples

# find where the label is 0 (cotton) and 1 (wool) in training / val  data
y_train_1=np.argwhere(y_train==1) # index where y_train=1 wool label
y_train_0=np.argwhere(y_train==0) # index where y_train=0 cotton label
y_train_2=np.argwhere(y_train==2) # index where y_train=0 cotton label

y_val_1=np.argwhere(y_val==1) # index where y_val=1 wool label
y_val_0=np.argwhere(y_val==0) # index where y_val=0 cotton label
y_val_2=np.argwhere(y_val==2) # index where y_val=0 cotton label

plt.figure(1)
plt.title('Data')
plt.plot(x_train.iloc[y_train_0[0,0]],'g', label='cotton') #values of absorbance in training data whose label is 0 - cotton
plt.plot(x_train.iloc[y_train_1[0,0]],'b', label='wool') #values of absorbance in training data whose label is 1 - wool
plt.plot(x_train.iloc[y_train_2[0,0]],'r', label='polyester') #values of absorbance in training data whose label is 1 - wool
plt.legend()
plt.show()

# %%
METRICS = [
      tf.keras.metrics.BinaryCrossentropy(name='cross entropy'),  # same as model's loss
      tf.keras.metrics.MeanSquaredError(name='Brier score'),
      tf.keras.metrics.TruePositives(name='tp'),
      tf.keras.metrics.FalsePositives(name='fp'),
      tf.keras.metrics.TrueNegatives(name='tn'),
      tf.keras.metrics.FalseNegatives(name='fn'), 
      tf.keras.metrics.BinaryAccuracy(name='accuracy'),
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
      tf.keras.metrics.AUC(name='auc'),
      tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
]

def make_model(metrics=METRICS):
  
  # model1 = tf.keras.Sequential([
  #     tf.keras.layers.Dense(
  #         16, activation='relu',
  #         input_shape=(x_train.shape[-1],)),
  #     tf.keras.layers.Dropout(0.5),
  #     tf.keras.layers.Dense(1, activation='sigmoid'),
  # ])

  model=tf.keras.Sequential([
    
  #tf.keras.layers.experimental.preprocessing.Rescaling(scale=1 / 127.5, input_shape=(n_row, n_col, 3), offset=-1),
  #tf.keras.layers.Dense(228, activation='relu',input_shape=(x_train.shape[-1],)),
  tf.keras.layers.Conv1D(8, kernel_size=8, input_shape=(x_train.shape[-1],1), strides=1,  activation='relu'),  
  tf.keras.layers.MaxPooling1D(pool_size=2),
  tf.keras.layers.Conv1D(16, 8, padding="same", activation="relu"),
  tf.keras.layers.MaxPooling1D(pool_size=2),
  
  tf.keras.layers.Conv1D(32, 8, padding="same", activation="relu"), 
  
  #tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
  #tf.keras.layers.MaxPooling1D(pool_size=2),
  #tf.keras.layers.Conv1D(64, 3, padding="same", activation="relu"),
  #tf.keras.layers.MaxPooling1D(pool_size=2),

  #tf.keras.layers.Conv1D(64, 3, activity_regularizer=tf.keras.regularizers.L2(0.01),padding="same", activation="relu"),
  tf.keras.layers.MaxPooling1D(pool_size=2),

  tf.keras.layers.Flatten(),

  # tf.keras.layers.Dense(64, 
  # activity_regularizer=tf.keras.regularizers.L2(0.01),
  # activation='relu'),

  tf.keras.layers.Dense(
      64, 
      #activity_regularizer=tf.keras.regularizers.L2(0.01),
      activation='relu'),
  tf.keras.layers.Dropout(0.2),
  #tf.keras.layers.Dense(1, activation='sigmoid'),
  tf.keras.layers.Dense(3, activation='softmax'),
    ])

  model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
        #loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), # use false when we have softmax at last layer
        metrics=metrics)

  return model

EPOCHS = 100 #50
BATCH_SIZE = 32

# early_stopping = tf.keras.callbacks.EarlyStopping(
#     monitor='val_prc', 
#     verbose=1,
#     patience=10,
#     mode='max',
#     restore_best_weights=True)

model = make_model()
model.summary()


# %%
log_path=r'C:\Users\devTe\Desktop\ML\log'
#ap_name=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#model_save_path=os.path.join(log_path,  ap_name + '.' + 'h5')

#log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
       #log_dir=log_dir, 
       histogram_freq=1,
       write_graph=False,
       write_images=False, #write model weights to visualize as image in TensorBoard.
       write_steps_per_second=False,
       update_freq='epoch', #'batch'
       profile_batch=0,
       embeddings_freq=0,
       embeddings_metadata=None,
       #**kwargs
)

#convert to one hot encoding

enc = OneHotEncoder()
#x_train_ohe = enc.fit_transform(np.asarray(x_train).astype('float32').reshape((-1,1))).toarray()
#x_val_ohe = enc.fit_transform(np.asarray(x_val).astype('float32').reshape((-1,1))).toarray() 
y_train_ohe = enc.fit_transform(np.asarray(y_train).astype('float32').reshape((-1,1))).toarray() 
y_val_ohe = enc.fit_transform(np.asarray(y_val).astype('float32').reshape((-1,1))).toarray()
y_test_ohe = enc.fit_transform(np.asarray(y_test).astype('float32').reshape((-1,1))).toarray()


history=model.fit(
 x_train,
 y_train_ohe,
 validation_data=(x_val, y_val_ohe),
 epochs=EPOCHS,
 verbose=1,
 callbacks=[tensorboard_callback]
)

# %%
#plot training history

def plot_metrics(history):
  metrics = ['loss', 'prc', 'precision', 'recall']
  colors=['b', 'g']
  for n, metric in enumerate(metrics):
    name = metric.replace("_"," ").capitalize()
    plt.subplot(2,2,n+1)
    plt.plot(history.epoch, history.history[metric], color=colors[0], label='Train')
    plt.plot(history.epoch, history.history['val_'+metric],
             color=colors[0], linestyle="--", label='Val')
    plt.xlabel('Epoch')
    plt.ylabel(name)
    if metric == 'loss':
      plt.ylim([0, plt.ylim()[1]])
    elif metric == 'auc':
      plt.ylim([0.8,1])
    else:
      plt.ylim([0,1])

    plt.legend()

plot_metrics(history)

# %%
#confusion matrix


test_predictions_baseline = model.predict(x_test, batch_size=BATCH_SIZE)
class_names=['Cotton', 'Wool', 'Polyester']
threshold=0.8

def plot_cm(labels, predictions, class_names, threshold):
  predictions > threshold
  np.argmax(predictions, axis=1)
  predicted_label=np.argmax(predictions, axis=1)
  
  cm = confusion_matrix(labels,predicted_label ) #predictions > threshold
  cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] #normalized cm
  
  plt.figure(figsize=(5,5))
  #sns.heatmap(cm, annot=True, fmt="d")
  #plt.subplot(2,1,1)
  #sns.heatmap(cmn, annot=True, fmt=".2f", xticklabels=class_names, yticklabels=class_names) 
 
  # plt.subplot(2,1,2)
  sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
  
  plt.title('Confusion matrix (Test Data set) @ {:.2f} threshold'.format(threshold))
  plt.ylabel('Actual label')
  plt.xlabel('Predicted label')

  print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
  print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
  print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
  print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
  print('Total Fraudulent Transactions: ', np.sum(cm[1]))

  print('Classification report : \n',classification_report(y_test, predicted_label, target_names=class_names)) #true_label, predicted_label

baseline_results = model.evaluate(x_val, y_val_ohe,
                                  batch_size=BATCH_SIZE, verbose=0)
for name, value in zip(model.metrics_names, baseline_results):
  print(name, ': ', value)
print()

plot_cm(y_test, test_predictions_baseline, class_names, threshold)




