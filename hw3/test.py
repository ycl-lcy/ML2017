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
from keras.models import load_model

pic_size = (48, 48)

test_data = pd.read_csv(sys.argv[1])
X_test = np.asarray(map(lambda f1: map(lambda f2: float(f2)/255, f1.split()), test_data['feature']))
X_test = X_test.reshape(X_test.shape[0], pic_size[0], pic_size[1], 1)

model = load_model("model")
ans = model.predict(X_test, verbose=2).argmax(axis=1)
ans = ans.reshape(1, ans.shape[0])
ans = pd.DataFrame({'label': ans[0]})
ans.index.name = 'id'
ans.to_csv(sys.argv[2])
