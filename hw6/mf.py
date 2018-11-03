from keras.models import Sequential, Model, load_model
from keras.layers import Input, Embedding, Dense, Dropout, Input, Activation, GRU, Flatten, Dot, Add
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.text import Tokenizer
from keras.regularizers import l2
import keras.backend as K

import pandas as pd
import numpy as np
import pickle 
import tensorflow as tf
import argparse
import sys

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

parser = argparse.ArgumentParser(description='Process the data and parameters.')
parser.add_argument('--train', type=int, default=0)
parser.add_argument('--data_path', default='data/')
parser.add_argument('--model_path', default='model/model.hdf5')
parser.add_argument('--ans_path', default='87')
parser.add_argument('--dim', type=int, default=20)
parser.add_argument('--epoch', type=int, default=10)
parser.add_argument('--nor', type=int, default=0)
args = parser.parse_args()

def get_model(n_users, n_items, latent_dim=args.dim):
    I1 = Input(shape=(1,))
    I2 = Input(shape=(1,))
    Ev1 = Embedding(n_users, latent_dim, embeddings_regularizer=l2(1e-8))(I1)
    Ev2 = Embedding(n_items, latent_dim, embeddings_regularizer=l2(1e-8))(I2)
    Ev1 = Flatten()(Ev1)
    Ev2 = Flatten()(Ev2)
    Eb1 = Embedding(n_users, 1, embeddings_regularizer=l2(1e-8))(I1)
    Eb2 = Embedding(n_items, 1, embeddings_regularizer=l2(1e-8))(I2)
    Eb1 = Flatten()(Eb1)
    Eb2 = Flatten()(Eb2)
    O = Dot(axes=1)([Ev1, Ev2])
    O = Add()([O, Eb1, Eb2])
    model = Model([I1, I2], O)
    model.summary()
    return model

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true - y_pred)))

#def train_model(model, X1_train, X2_train, Y_train):

train = pd.read_csv(args.data_path + "train.csv")
train = train.sample(frac=1).reset_index(drop=True)
X1_train = train['UserID'].astype(str)
X2_train = train['MovieID'].astype(str)
Y_train = train['Rating']
if args.nor == 1:
    Y_train_mean = Y_train.mean()
    Y_train_std = Y_train.std()
    Y_train = (Y_train - Y_train.mean())/(Y_train.std())
test = pd.read_csv(args.data_path + "test.csv")
X1_test = test['UserID'].astype(str)
X2_test = test['MovieID'].astype(str)


users = pd.read_csv(args.data_path + "users.csv", sep='::', engine='python') 
movies = pd.read_csv(args.data_path + "movies.csv", sep='::', engine='python')

userID = users['UserID'].astype(str)
movieID = movies['movieID'].astype(str)


model = get_model(6041, 3884)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=[rmse])

if args.train == 1:
    tokenizer_users = Tokenizer()
    tokenizer_users.fit_on_texts(userID)
    X1_train = tokenizer_users.texts_to_sequences(X1_train)
    X1_train = np.asarray(X1_train)
    n_users = len(tokenizer_users.word_index)+1
    with open("mapping/mapping_users", 'wb') as file:
        pickle.dump(tokenizer_users.word_index, file)

    tokenizer_items = Tokenizer()
    tokenizer_items.fit_on_texts(movieID)
    X2_train = tokenizer_items.texts_to_sequences(X2_train)
    X2_train = np.asarray(X2_train)
    n_items = len(tokenizer_items.word_index)+1
    with open("mapping/mapping_items", 'wb') as file:
        pickle.dump(tokenizer_items.word_index, file)
    
    checkpoint = ModelCheckpoint(filepath=args.model_path,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_rmse',
                                     mode='min')
    model.fit([X1_train, X2_train], 
              Y_train, 
              epochs=args.epoch,
              batch_size=1024,
              validation_split=0.2,
              callbacks=[checkpoint])
else:
    tokenizer_users = Tokenizer()
    with open("mapping/mapping_users", 'rb') as file:
        tokenizer_users.word_index = pickle.load(file)
    
    tokenizer_items = Tokenizer()
    with open("mapping/mapping_items", 'rb') as file:
        tokenizer_items.word_index = pickle.load(file)
    
    model.load_weights(args.model_path) #custom_objects={':rmse': rmse})

#user_emb = np.array(model.layers[2].get_weights()).squeeze()
#movie_emb = np.array(model.layers[3].get_weights()).squeeze()
#np.savetxt("user_emb.npy", user_emb)
#np.savetxt("movie_emb.npy", movie_emb)

X1_test = tokenizer_users.texts_to_sequences(X1_test)
X1_test = np.array(X1_test)

X2_test = tokenizer_items.texts_to_sequences(X2_test)
X2_test = np.array(X2_test)

Y_test = model.predict([X1_test, X2_test])
Y_test = Y_test.reshape(Y_test.shape[0])
if args.nor == 1:
    Y_test = (Y_test * Y_train_std) + Y_train_mean
ans = pd.DataFrame({'TestDataID': test['TestDataID'].as_matrix(), 'Rating': Y_test})
ans = ans[['TestDataID', 'Rating']]
#ans = ans.sort_values('TestDataID')
ans.to_csv(args.ans_path, index=False)

#X2_test = X2_test.reshape(X2_test.shape[0], 1)

#print (len(tokenizer_users.word_index))
# map_users = {None: None}
# map_items = {None: None}
# n_users = 0
# n_items = 0
# for i in range(len(train)):
    # if X1_train[i] not in map_users:
        # map_users[X1_train[i]] = n_users
        # X1_train[i] = n_users
        # n_users += 1
    # else:
        # X1_train[i] = map_users[X1_train[i]]

    # if X2_train[i] not in map_items:
        # map_items[X2_train[i]] = n_items
        # X2_train[i] = n_items
        # n_items += 1
    # else:
        # X2_train[i] = map_items[X2_train[i]]
# print (n_users)
# print (n_items)
#print (train['UserID'].value_counts())
#print (train['MovieID'].value_counts())
#print (train['Rating'].value_counts())


# model = Model(I, E)
# model.compile(loss='mse', optimizer='sgd')
# model.summary()

# I = Input((300, ))
# E = Embedding(input_dim=5000, output_dim=100, trainable=False)(I)
# R = GRU(256, activation='tanh', dropout=0.5, recurrent_dropout=0.3)(E)
# D = Dense(units=256, kernel_initializer="glorot_normal")(R)
# D = Activation('relu')(D)
# D = Dropout(0.25)(D)
# D = Dense(128, kernel_initializer="glorot_normal")(D)
# D = Activation('relu')(D)
# D = Dropout(0.25)(D)
# D = Dense(64, kernel_initializer="glorot_normal")(D)
# D = Activation('relu')(D)
# D = Dropout(0.5)(D)
# O = Dense(units=38, activation='sigmoid', kernel_initializer="glorot_normal")(D)
# model = Model(inputs=I, outputs=O)
# model.summary()
