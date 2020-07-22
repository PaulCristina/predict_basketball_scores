#------------------------------------------------------------------------------------------------------------#
### 1.0 LOAD LIBRARY'S ----
import os, sys
import pandas as pd
import numpy as np
import dateutil

from keras.models import Sequential
from keras.layers import Dense

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

### Get model predicions 
model.predict(x_test)
test

print("Check weight: {}".format(model.get_weights()))
