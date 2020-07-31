#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
from __future__ import absolute_import, division, print_function

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
# 1.Create the training and testing datasets.
# Typically, training is either 70% or 80% of the 
# total dataset.
# 2. Train the neural network against your training data and 
# validate against your testing data.
# 3. Graph the results to validate that you are getting 
# closer to your desired results.

# Load the data set from the last milestone 1
column_names = ['Date','HomeTeam','HomeScore','AwayTeam','AwayScore',
                'HomeScoreAverage','HomeDefenseAverage','AwayScoreAverage','AwayDefenseAverage',
                'Result']

games_csv = 'https://liveproject-resources.s3.amazonaws.com/other/deeplearningbasketballscores/Games-Calculated.csv'
all_data = pd.read_csv(games_csv, header=None, names=column_names)
all_data.head()

#------------------------------------------------------------------------------------------------------------#
### 2.0 SPLIT DATASET ----

# Create the Train/Test/Validate data sets
# All about data

# When training a network we need to send in data for it to learn. We can't then use the 
# same data to test if it is learning. It would be like working a problem in school and 
# then get that same problem on the test. All it proves is you know that data. We need 
# the model to generalize the data versus knowing the actual data.

# Generalization is a term used to describe the model's ability to understand new data 
# it hasn't seen before. In this project, we need to generalize to all games in the future
#  versus the individual games in the past. When the model doesn't generalize it gets 
#  into a scenario where it overfits.

# Overfitting is a term used to describe when the model only learns the data it has 
# versus the new data. You can see this visually in the graph of the errors resulting 
# from training. You will see that your training errors will decrease while your 
# validation errors will start to increase again.
# Splits

# As stated above, we need to ensure we don't overfit to the data. To handle this we 
# need to separate the data into 3 data sets (With the way Keras handles the validation 
#                                             within the network we only create 2). 
# In my example, I used 80% of train (which will get split 80/20 for validation) and 
# 20% for testing. This allows me plenty of randomized data to test my model.
# Data Labels

# You will notice that I remove the Results column and name those as the labels. The 
# labels are used as the answers or truth in the network. When the network gets the 
# data it will then try and predict its own label and compare it versus the correct 
# label. After they get the error (we use mean square error) it uses an algorithm 
# called backpropagation to go back through all the weights and biases in the nodes 
# to attempt to get the answer correct next time.
# Data Normalization

# At the end of the code I normalize all of the data. The reason for me doing this 
# is to ensure that the scale of the values are all similiar. If our first input 
# ranges from 30 to 120 and the second input ranges from 2 to 10 the first input 
# will have an outsized impact on our learning.

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

# # Normalize the data
# mean = train.mean(axis=0)
# train_data = train - mean
# std = train_data.std(axis=0)
# train_data /= std

# test_data = test - mean
# test_data /= std

# train_data.describe()
# test_data.describe()

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

# Create a sequential neural network with 3 dense layers, and compile
# Parameters

# Sequential (Documantaion): A sequential network runs linearly 
# through the layers.
# Dense (Documentaion): A densely connected network layer. This is the standard
#  for most neural networks.
# Neurons/Units: We set these to 32 to start. This is the dimensionality of the 
# output space. Since we just want a final score differences we set the last 
# value to 1.
# Input Shape: This is the size of the data you are using to train. In our case, 
# we have home team score, away team score, home team defense and away team 
# defense. So, we set this value to 4.
# Activation (Documentation): We picked Recified Linear Unit (relu). Using trial 
# and error you can determine which one is best for your data. There are some 
# activations that are for specific network results.
# Optimizers (Documentation): Optimizers are used during training to try and 
# find the global optimum which will lead to better results. We chose RMSProp 
# as a general purpose optimizer.
# Metrics (Documentation): This array determines what values are tracked and 
# returned during training. We will use these to graph our results and determine 
# how well our model is representing our data.
# Loss Function (Documentation): The loss function is what is computed to determine
#  how good or bad our output matches our expected results.

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

bballmodel = Build_Model()

# This method will be used in place of the normal output. This is cleaner
 # in my opinion
class PrintDoc(keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs):
    if epoch % 10 == 0: print('')
    print('.', end='')
    
# Model Complexity

# I am going to take this section to talk about the complexity of the model
#  as well as high bias and high variance.

# Bias: A model with low model complexity with a high error rate is said to have
#  high bias. The high bias comes from underfitting the data.

# Variance: A model with high complexity with a high error rate is said to have 
# high variance. The high variance comes from overfitting the data.

#------------------------------------------------------------------------------------------------------------#
### 4.0 TRAIN MODEL ----

history = bballmodel.fit(train_data, 
                         train_labels, 
                         epochs=100, 
                         validation_split=0.2, 
                         verbose=True
                         # ,  callbacks=[PrintDoc()]
                         )


# Check the results
# Create a DataFrame from the output from the fit method
hist = pd.DataFrame(history.history)
# Create an epoch column and set it to the epoch index
hist['epoch'] = history.epoch
hist.tail()

# Test the network against your testing data set
test_loss, mae, test_acc, mse = bballmodel.evaluate(test_data, test_labels)
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
