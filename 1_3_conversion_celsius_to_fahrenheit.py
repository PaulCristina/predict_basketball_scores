#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
import os, sys
import pandas as pd
import numpy as np
import dateutil



# Workflow
# 1.Create a neural network that can convert Celsius to Fahrenheit 
# and examine the weights.
# 2. Using TensorFlow andKeras build a single layered network with a
#  single input and a single output and a single neuron.
# 3. Using NumPy generate ~20 input/output values to test
# 4. Train the neural network on your data.
# 5. Test the neural network to see if it is working.
# 6. Examine the weights to see if they match F = C * 1.8 + 32F=C∗1.8+32.

### train dataframe
num = np.random.randint(-100, 100, size=(20, 1))
train = pd.DataFrame(num, columns=['degrees_c'])
train['degrees_f'] = train['degrees_c'] * 1.8 + 32

### test dataframe
num = np.random.randint(-100, 100, size=(5, 1))
test = pd.DataFrame(num, columns=['degrees_c'])
test['degrees_f'] = test['degrees_c'] * 1.8 + 32
print(train)

### split into dependent/independent predictors
x_train = train.drop(['degrees_f'], axis=1)
y_train = train['degrees_f']


x_test = test.drop(['degrees_f'], axis=1)
y_test = test['degrees_f']

#------------------------------------------------------------------------------------------------------------#
### 2.0 KERAS ----

from keras.models import Sequential
from keras.layers import Dense

# Create a Sequential model
model = Sequential()

# Need ~50 neurons in order for the predictions to be accurate
# Add dense layers 
model.add(Dense(50, input_shape=(1,)))

# Add two Dense layers with 50 neurons
model.add(Dense(50))
model.add(Dense(50))

# End your model with a Dense layer and no activation
model.add(Dense(1))

# Compile your model
model.compile(optimizer="adam", loss="mse")

# Fit your model on your data for 30 epochs
model.fit(x_train, y_train, epochs = 500)

# Get model predicions 
model.predict(x_test)
test

print("Check weight: {}".format(model.get_weights()))


#------------------------------------------------------------------------------------------------------------#
### 3.0 TENSORFLOW ----

import tensorflow as tf
print(tf.__version__)

# Below are the key parts to the network:
# - Sequential: We will want this to be a sequential network. For the most part, 
# this is the default type. It just means that the data flows sequentially 
# through all of the layers.
# - Dense: This is the simplest layer available. For a deeper understanding you 
# can check out the official documentation here
# units: This specifies the number of neurons in the layer. In other words, this
#  is the number of variables the layer has to learn.
# input_shape: This specifies how many parameters that we will pass to our 
# network. Since we are just going to send in the temperate in Celcius 
# we only need 1.
# - compile: We need to compile the network to be able to start using it.
# - loss: This is the loss function. It is how the network is able to determin 
# how far off the prediction is from the desired outcome.
# - optimizer: This determines the way the internal values are adjusted to reduce 
# the loss. For more information on the Adam optimizer go to the documentation
#  here. 

model = tf.keras.Sequential(
    tf.keras.layers.Dense(units=1, input_shape=[1])
)
model.compile(loss='mean_squared_error', 
              optimizer=tf.keras.optimizers.Adam(0.1))

# Below are the key parts of the training code:
# - history: These values are the results of the 
# training. We will use these after to graph what 
# we have done.
# - fit: This is the method that does the training of 
# the model. We are passing in our celsius data and 
# also passing in our expected output (fahrenheit)
#  data.
# - epochs: This parameter specifies how many times 
# this cycle should be run. 
history = model.fit(x_train, y_train, epochs=500, verbose=True)

import matplotlib.pyplot as plt
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.plot(history.history['loss'])


# Get model predicions 
model.predict(x_test)
test

# Examine weights
print("This is the weight that should be pretty close to the *1.8 in the formula: {}".format( model.layers[0].get_weights()[0][0] ))
print("This is the bias that should be pretty close to the +32 in the formula: {}".format( model.layers[0].get_weights()[1] ))

