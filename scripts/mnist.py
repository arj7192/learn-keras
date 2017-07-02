# all imports here

import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.datasets import mnist


# always good to seed any randomness

np.random.seed(123)


# load, preprocess data

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
Y_train = np_utils.to_categorical(y_train, 10)
Y_test = np_utils.to_categorical(y_test, 10)


# model definition

model = Sequential()
model.add(Convolution2D(32, (3, 3), activation='relu', input_shape=(28,28,1)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))


# model compilation

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# model training

model.fit(X_train, Y_train, 
          batch_size=32, nb_epoch=1, verbose=1)


# model evaluation

score = model.evaluate(X_test, Y_test, verbose=1)
print '\n'
print model.metrics_names
print score
