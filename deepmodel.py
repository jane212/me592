

from __future__ import print_function
import os
import numpy as np
dirpath = '/home/microway/Tingting/'
dat = np.loadtxt(dirpath+"dat_norm.txt",delimiter=",")

X = dat[:,1:]
y = dat[:,0]

from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
x_test, x_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.5)


import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation
from keras.layers.normalization import BatchNormalization

batch_size = 2000
num_classes = 11
epochs = 30

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
y_val = keras.utils.to_categorical(y_val, num_classes)

input_shape = (x_train.shape[1],)

model = Sequential()
model.add(Dense(40,input_shape=input_shape))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(33))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(26))
model.add(BatchNormalization())
model.add(Activation('relu'))
#model.add(Dropout(0.2))
model.add(Dense(18))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(x_val, y_val))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])




