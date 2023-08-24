import tensorflow as tf
import tensorflow_hub as tfhub
from google.colab import files
import os
import shutil
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def auth_kaggle():
  # authenticate on kaggle
  print('Please upload the kaggle.json file.')

  # upload the kaggle.json file
  uploaded = files.upload()

  for fn in uploaded.keys():
    print('User uploaded file "{name}" with length {length} bytes'.format(
        name=fn, length=len(uploaded[fn])))
    
  # Then move kaggle.json into the folder where the API expects to find it.
  os.system('mkdir -p ~/.kaggle/ && mv kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json')
  return

def download_data():
  # download and extract data
  os.system('kaggle datasets download "irmiot22/fsc22-dataset"')
  os.system('unzip fsc22-dataset.zip')

  # rename data folders
  os.rename('Audio Wise V1.0-20220916T202003Z-001','raw_data')
  os.rename('raw_data/Audio Wise V1.0', 'raw_data/audio')
  os.rename('Metadata-20220916T202011Z-001/Metadata/Metadata V1.0 FSC22.csv','raw_data/label.csv')

  # delete unused stuff
  shutil.rmtree('Metadata-20220916T202011Z-001')
  os.remove('fsc22-dataset.zip')
  return

def data_loading():

  auth_kaggle()
  download_data()

  # Load the model.
  yamnet = tfhub.load('https://tfhub.dev/google/yamnet/1')

  # load labels
  lab = pd.read_csv('raw_data/label.csv')

  # renormalize to 0 class IDs
  lab['Class ID'] = lab['Class ID'] - 1

  # load and resample data
  # it takes about 6 minutes
  data = list(map(lambda x: librosa.load('raw_data/audio/'+x, sr = 16000)[0], lab['Dataset File Name'].to_list()))

  # normalize to have data in [-1,1] (to be fed to Yamnet)
  data = tf.constant(librosa.util.normalize(data))

  # extract the spectrograms used in YAMNet
  processed_data = list(map(lambda x: yamnet(x)[2], data))

  # build the dataset
  X = np.array(processed_data)
  y = lab['Class ID'].to_numpy()




  # split in train, validation and test
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=135, random_state=69420,stratify=y)
  X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=405, random_state=69,stratify=y_train)

  # convert to tensors
  X_train = tf.expand_dims(tf.constant(X_train),-1)
  y_train = tf.constant(y_train)

  X_val = tf.expand_dims(tf.constant(X_val),-1)
  y_val = tf.constant(y_val)

  X_test = tf.expand_dims(tf.constant(X_test),-1)
  y_test = tf.constant(y_test)

  # generate tfdataset object
  train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(3000).batch(48)
  val_data = tf.data.Dataset.from_tensor_slices((X_val, y_val)).shuffle(3000).batch(32)
  test_data = tf.data.Dataset.from_tensor_slices((X_test, y_test)).shuffle(3000).batch(32)

  return(train_data,val_data, test_data)