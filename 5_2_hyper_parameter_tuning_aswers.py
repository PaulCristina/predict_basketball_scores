#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
from __future__ import absolute_import, division, print_function

from tensorflow import keras
from tensorflow.keras import layers
from kerastuner.tuners import RandomSearch

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from sklearn.preprocessing import StandardScaler



pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)


np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

seed(42)
tensorflow.random.set_seed(42)


# Workflow

# Start with your original network and check the mean absolute error that 
# you get from running the evaluate method on your testing data.
# Create a series of networks that try and beat those results. Check the 
# notes section for a recommendation on the process.
# Alter the network’s layers, neurons, optimizers, activation functions, 
# loss function, and so on.
# Repeat until you have a satisfactory result.
# Download your best model in h5 format.

# Load the data set from the last milestone 1
column_names = ['Date','HomeTeam','HomeScore','AwayTeam','AwayScore',
                'HomeScoreAverage','HomeDefenseAverage','AwayScoreAverage','AwayDefenseAverage',
                'Result']

games_csv = 'https://liveproject-resources.s3.amazonaws.com/other/deeplearningbasketballscores/Games-Calculated.csv'
all_data = pd.read_csv(games_csv, header=None, names=column_names)
all_data.head()

#------------------------------------------------------------------------------------------------------------#
### 2.0 SPLIT DATASET ----

# Drop the columns that we are NOT going to train on
all_data.drop(['Date','HomeTeam','HomeScore','AwayTeam','AwayScore'], 
              axis=1, inplace=True)
all_data.tail()

#Break it into 80/20 splits
train = all_data.sample(frac=0.8, random_state=0)
test = all_data.drop(train.index)
print('Training Size: %s' % train.shape[0])
print('Testing Size: %s' % test.shape[0])

#Create the labels
train_labels = train.pop('Result')
test_labels = test.pop('Result')

# Standardize the train and test features
sc = StandardScaler()
train_data = pd.DataFrame(sc.fit_transform(train))
test_data = pd.DataFrame(sc.transform(test))

train_data.columns = train.columns
test_data.columns = test.columns

train_data.describe()
test_data.describe()

#------------------------------------------------------------------------------------------------------------#
### 3.0 NEURAL NETWORK----

# Milestone 3 Network
def Build_Model_Milestone3():
  model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=[4]),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.RMSprop()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

m3 = Build_Model_Milestone3()
m3_history = m3.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=0)


_, m3_mean_absolute_error, _, _ = m3.evaluate(test_data, test_labels,verbose=0)
print('Milestone 3 model: %s' % m3_mean_absolute_error)

# Less Neutrons
# I started with 32 neutrons in each layer. I am going to adjust that down to 24, 12, and 8.
def Build_Model_24Neutrons():
  model = keras.models.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=[4]),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.RMSprop()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

def Build_Model_12Neutrons():
  model = keras.models.Sequential([
    keras.layers.Dense(12, activation='relu', input_shape=[4]),
    keras.layers.Dense(12, activation='relu'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.RMSprop()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

def Build_Model_8Neutrons():
  model = keras.models.Sequential([
    keras.layers.Dense(8, activation='relu', input_shape=[4]),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.RMSprop()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

# Train the networks
m8 = Build_Model_8Neutrons()
m8_history = m8.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

m12 = Build_Model_12Neutrons()
m12_history = m12.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

m24 = Build_Model_24Neutrons()
m24_history = m24.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

# Grab Their Results
_, m8_mean_absolute_error, _, _ = m8.evaluate(test_data, test_labels,verbose=0)
print('8 Neurons model: %s' % m8_mean_absolute_error)

_, m12_mean_absolute_error, _, _ = m12.evaluate(test_data, test_labels,verbose=0)
print('12 Neurons model: %s' % m12_mean_absolute_error)

_, m24_mean_absolute_error, _, _ = m24.evaluate(test_data, test_labels,verbose=0)
print('24 Neurons model: %s' % m24_mean_absolute_error)

print('Milestone 3 model: %s' % m3_mean_absolute_error)

# Activation Functions
# We currently have RELU (rectified linear unit). We will try sigmoid and softmax.

def Build_Model_Sigmoid():
  model = keras.models.Sequential([
    keras.layers.Dense(24, activation='sigmoid', input_shape=[4]),
    keras.layers.Dense(24, activation='sigmoid'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.RMSprop()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

def Build_Model_Softmax():
  model = keras.models.Sequential([
    keras.layers.Dense(24, activation='softmax', input_shape=[4]),
    keras.layers.Dense(24, activation='softmax'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.RMSprop()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

# Train the networks
msg = Build_Model_Sigmoid()
msg_history = msg.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

msm = Build_Model_Softmax()
msm_history = msm.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

_, msg_mean_absolute_error, _, _ = msg.evaluate(test_data, test_labels,verbose=0)
print('Sigmoid model: %s' % msg_mean_absolute_error)

_, msm_mean_absolute_error, _, _ = msm.evaluate(test_data, test_labels,verbose=0)
print('Softmax model: %s' % msm_mean_absolute_error)


print('24 Neuron model: %s' % m24_mean_absolute_error)

# Optimizers
# We started with RMSProp and we are going to try SDG (stochastic gradient descent), Adam, and Adamax.

def Build_Model_SDG():
  model = keras.models.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=[4]),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.SGD()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

def Build_Model_Adam():
  model = keras.models.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=[4]),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.Adam()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

def Build_Model_Adamax():
  model = keras.models.Sequential([
    keras.layers.Dense(24, activation='relu', input_shape=[4]),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1)                                   
  ])
  
  opt = keras.optimizers.Adamax()
  m = [
       keras.metrics.MeanAbsoluteError(),
       keras.metrics.Accuracy(),
       keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss=l, optimizer=opt, metrics=m)
  return model

# Train the networks
sdg = Build_Model_SDG()
sdg_history = sdg.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

adam = Build_Model_Adam()
adam_history = adam.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

ada = Build_Model_Adamax()
ada_history = ada.fit(train_data, train_labels, epochs=100, validation_split=0.2, verbose=True)

_, sdg_mean_absolute_error, _, _ = sdg.evaluate(test_data, test_labels,verbose=0)
print('SDG model: %s' % sdg_mean_absolute_error)

_, adam_mean_absolute_error, _, _ = adam.evaluate(test_data, test_labels,verbose=0)
print('Adam model: %s' % adam_mean_absolute_error)

_, ada_mean_absolute_error, _, _ = ada.evaluate(test_data, test_labels,verbose=0)
print('Adamax model: %s' % ada_mean_absolute_error)


print('RMSProp model: %s' % m24_mean_absolute_error)

# Export h5 Model

# We will use the built in Keras save function to export the model
#Save the model and all it's weights
m24.save('C:/Users/Guest01/Documents/github_projects/predict_basketball_scores/bball/deeplearning-manning.h5')

# #Google Colab code to download
# from google.colab import files
# files.download('deeplearning-manning.h5')
