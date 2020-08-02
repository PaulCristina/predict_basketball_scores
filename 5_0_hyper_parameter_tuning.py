#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
from __future__ import absolute_import, division, print_function

import tensorflow as tf
import tensorflow.keras as keras

# Import KerasClassifier from keras scikit learn wrappers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

# Import the early stopping callback
from keras.callbacks import EarlyStopping

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler



pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)


np.set_printoptions(suppress=True)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

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
### 3.0 NEURAL NETWORK GRID----

# Create hyperparameter options
# Define the parameters to try out
params = {'activation': ['relu', 'tanh'], 
          'batch_size': [16, 32, 64], 
          'epochs': [50], 
          'learning_rate': [0.0001, 0.001, 0.01, 0.1],
          'neurons':[16, 32]}

def Build_Model(learning_rate, activation, neurons):
    # Start neural network
  model = keras.models.Sequential([
    # Add fully connected layer with a XXX activation function
    keras.layers.Dense(neurons, activation = activation, input_shape=(train_data.shape[1],)),
    # Add fully connected layer with a XXX activation function
    keras.layers.Dense(neurons, activation = activation),
    # Add fully connected layer with a XXX activation function
    keras.layers.Dense(1)                                   
  ])

  opt = keras.optimizers.Adam(learning_rate=learning_rate)  
  
  m = [
        keras.metrics.MeanAbsoluteError(),
        keras.metrics.Accuracy(),
        keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss = l, optimizer = opt, metrics = m)
  return model

# Create a KerasClassifier
model = KerasClassifier(build_fn = Build_Model)

# Create grid search
grid = GridSearchCV(estimator=model, 
                    cv=3, 
                    param_grid=params)

# Fit grid search
grid_result = grid.fit(train_data, train_labels)

# View hyperparameters of best neural network
grid_result.best_params_

# Print results
print("Best score: {}".format(grid_result.best_score_ ))
print("Best parameters: {}".format(grid_result.best_params_))

#------------------------------------------------------------------------------------------------------------#
### 3.0 NEURAL NETWORK WITH BEST RESULTS----

# Define a callback to monitor val_acc
monitor_val_loss = EarlyStopping(monitor='val_loss', 
                       patience=5)

# Create hyperparameter options
# Define the parameters to try out
params = {'activation': ['relu'], 
          'batch_size': [16], 
          'epochs': [50], 
          'learning_rate': [0.0001],
          'neurons':[16]}

def Build_Model(learning_rate, activation, neurons):
    # Start neural network
  model = keras.models.Sequential([
    # Add fully connected layer with a XXX activation function
    keras.layers.Dense(neurons, activation = activation, input_shape=(train_data.shape[1],)),
    # Add fully connected layer with a XXX activation function
    keras.layers.Dense(neurons, activation = activation),
    # Add fully connected layer with a XXX activation function
    keras.layers.Dense(1)                                   
  ])

  opt = keras.optimizers.Adam(learning_rate=learning_rate)  
  
  m = [
        keras.metrics.MeanAbsoluteError(),
        keras.metrics.Accuracy(),
        keras.metrics.MeanSquaredError()
  ]
  l = keras.losses.MeanSquaredError()
  
  model.compile(loss = l, optimizer = opt, metrics = m)
  return model

model = Build_Model(activation = 'relu', learning_rate = 0.0001, neurons = 16)

history = model.fit(train_data, 
                         train_labels, 
                         epochs=50, 
                         validation_split=0.2, 
                         verbose=True
                         # ,callbacks= [monitor_val_loss]
                         )

# Test the network against your testing data set
test_loss, mae, test_acc, mse = model.evaluate(test_data, test_labels)
mae

def plot_history(history):
  plt.figure()
  plt.xlabel('Epoch')
  plt.ylabel('Mean Absolute Error')
  plt.plot(history['epoch'], history['mean_absolute_error'],
           label='Train Mean Absolute Error')
  plt.plot(history['epoch'], history['val_mean_absolute_error'],
           label = 'Val Mean Absolute Error')
  plt.legend()
  #plt.ylim([0,1])
  plt.show()
  
# Check the results
# Create a DataFrame from the output from the fit method
hist = pd.DataFrame(history.history)
# Create an epoch column and set it to the epoch index
hist['epoch'] = history.epoch
hist.tail()

plot_history(hist)

