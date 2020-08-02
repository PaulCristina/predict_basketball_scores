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
def build_model(hp):
    model = keras.Sequential()
    for i in range(hp.Int('num_layers', 1, 6)):
        model.add(layers.Dense(units=hp.Int('units_' + str(i),
                                            min_value = 8,
                                            max_value = 32,
                                            step = 4),
                                activation = hp.Choice('dense_activation', 
                                                       values = ['relu', 'tanh', 'sigmoid'])))
    model.add(layers.Dropout(hp.Float('dropout', min_value=0.01, max_value=0.2,step=0.02)))    
    model.add(layers.Dense(1))
    model.compile(
        optimizer = keras.optimizers.Adam(
            hp.Choice('learning_rate', [0.0001, 0.001, 0.01, 0.1])),
        loss = keras.losses.MeanSquaredError(),
        metrics = keras.metrics.MeanAbsoluteError())
    return model

tuner = RandomSearch(
    build_model,
    objective = 'val_loss',
    max_trials = 10,
    executions_per_trial = 3,
    overwrite = True
    )

tuner.search_space_summary()

tuner.search(train_data, train_labels,
             epochs = 20,
             validation_data=(test_data, test_labels))

tuner.results_summary()

# get bets model
best_model = tuner.get_best_models(num_models=1)[0]

# get the optimal hyperparameters
best_hps = tuner.get_best_hyperparameters(num_trials = 1)[0]

print(f"""
The hyperparameter search is complete. 
- optimal number of units in the 0 layer is {best_hps.get('units_0')} 
- optimal number of units in the 1 densely-connected layer is {best_hps.get('units_1')} 
- optimal number of units in the 2 densely-connected layer is {best_hps.get('units_2')} 
- optimal number of units in the 3 densely-connected layer is {best_hps.get('units_3')} 
- optimal number of units in the 4 densely-connected layer is {best_hps.get('units_4')}
- optimal number of units in the 5 densely-connected layer is {best_hps.get('units_5')}
- optimal activation is is {best_hps.get('dense_activation')} 
- optimal dropout is {best_hps.get('dropout')} 
- the optimal learning rate for the optimizer is {best_hps.get('learning_rate')}.
""")

# Build the model with the optimal hyperparameters and train it on the data
model = tuner.hypermodel.build(best_hps)

history = model.fit(train_data, 
                    train_labels, 
                    epochs=20, 
                    validation_data=(test_data, test_labels), 
                    verbose=True
                    # ,callbacks= [monitor_val_loss]
                    )

model.summary()

# Test the network against your testing data set
mae, mse = model.evaluate(test_data, test_labels)

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