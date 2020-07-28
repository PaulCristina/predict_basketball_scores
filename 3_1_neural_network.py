#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----

import os
import sys
import pandas as pd
import numpy as np

pd.set_option('display.max_rows', 10)
pd.set_option('display.max_columns', 10)
pd.set_option('display.width', 100)

# Import the sequential model and dense layer
from keras.models import Sequential
from keras.layers import Dense
from keras import models
from tensorflow import keras
from tensorflow.keras import layers

# Import seaborn
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

# Workflow
# 1.Create the training and testing datasets.
# Typically, training is either 70% or 80% of the 
# total dataset.
# 2. Train the neural network against your training data and 
# validate against your testing data.
# 3. Graph the results to validate that you are getting 
# closer to your desired results.



### set header for datasets
columns_games_df = ['Date','HomeTeam','HomeScore','AwayTeam','AwayScore',
                     'HomeScoreAverage', 'HomeDefenseAverage', 'AwayScoreAverage', 'AwayDefenseAverage', 
                     'Result']

### change `DATA_DIR` to the location where movielens-20m dataset sits
DATA_DIR = 'C:/Users/Guest01/Documents/github_projects/predict_basketball_scores/data'
df_games = pd.read_csv(os.path.join(DATA_DIR, 'Games-Calculated.csv'),
                       header=None, names=columns_games_df)
print(df_games)

#------------------------------------------------------------------------------------------------------------#
### 2.0 SPLIT DATASET ----
train, test = train_test_split(df_games, 
                               test_size=0.2,
                               random_state=1234)

y_train = train.loc[:, ['Result']]
x_train = train.loc[:, ['HomeScoreAverage', 'HomeDefenseAverage', 'AwayScoreAverage',
                        'AwayDefenseAverage']]

y_test = test.loc[:, ['Result']]
x_test = test.loc[:, ['HomeScoreAverage', 'HomeDefenseAverage', 'AwayScoreAverage',
                        'AwayDefenseAverage']]

#------------------------------------------------------------------------------------------------------------#
### 3.0 NEURAL NETWORK ----

# Create a sequential model
model = Sequential()

# Add dense layers
model.add(Dense(16, input_shape=(x_train.shape[1],), activation = 'relu'))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(4, activation = 'relu'))

model.add(Dense(1, activation = 'relu'))
model.summary()

# Compile your model
model.compile(optimizer="adam", 
              loss="mse",
              metrics=['mae'])

# Train the model
history = model.fit(x = x_train, 
                    y = y_train, 
                    validation_split=0.2,
	                epochs = 100, 
                    batch_size = 32,
                    shuffle=True,
                    verbose = True)

print(history.history.keys())


# Plot train vs test loss per epoch
plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Plot train vs test mae per epoch
plt.figure()
plt.plot(history.history['mae'])
plt.plot(history.history['val_mae'])

plt.title('Model mae')
plt.ylabel('Mae')
plt.xlabel('Epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# Evaluate your model's accuracy on the test data
mae = model.evaluate(x_test, y_test)[1]
mae
