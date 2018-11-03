import numpy as np
from numpy import genfromtxt
import pandas as pd
from random import shuffle
import sys
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras.datasets import mnist
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

pic_size = (48, 48)

def train():
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_split=0.2, verbose=1)
    model.save("model")

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

train_data = pd.read_csv(sys.argv[1])
X_train = map(lambda f1: map(lambda f2: float(f2)/255, f1.split()), train_data['feature'])
#shuffle(X_train)
X_train = np.asarray(X_train)
X_train = X_train.reshape(X_train.shape[0], pic_size[0], pic_size[1], 1)
Y_train = np.asarray(train_data['label'])
Y_train = np_utils.to_categorical(Y_train, 7)

model = Sequential()

#model.add(Convolution2D(32, (3, 3), input_shape=(pic_size[0], pic_size[1], 1)))
#model.add(Activation('relu'))
model.add(Convolution2D(32, (3, 3), input_shape=(pic_size[0], pic_size[1], 1), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Convolution2D(64, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Convolution2D(128, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Convolution2D(256, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Convolution2D(512, (3, 3), padding='same', activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.5))

model.add(Flatten())

model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))

model.summary()

train()
