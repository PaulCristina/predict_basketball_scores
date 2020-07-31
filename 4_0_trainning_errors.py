#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
import tensorflow as tf
from tensorflow import keras

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

# Overfit a the neural network from milestone 3.
# There are ways to overfit your network, but for this 
# milestone we want to over-train on the training data. 
# See Chapter 6 of Grokking Deep Learning for more details.
# Graph the training versus validation metrics prove overfitting.

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
### 3.0 NEURAL NETWORK ----


def Build_Model():
  model = keras.models.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(train_data.shape[1],)),
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

model = Build_Model()

 

#------------------------------------------------------------------------------------------------------------#
### 4.0 TRAIN MODEL ----

history = model.fit(train_data, 
                         train_labels, 
                         epochs=300, 
                         validation_split=0.2, 
                         verbose=True
                         )


# Check the results
# Create a DataFrame from the output from the fit method
hist = pd.DataFrame(history.history)
# Create an epoch column and set it to the epoch index
hist['epoch'] = history.epoch
hist.tail()

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

plot_history(hist)
