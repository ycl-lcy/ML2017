import numpy as np
import string
import sys
import keras.backend as K 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, Model
from keras.layers import Dense,Dropout,Input,Activation
from keras.layers import GRU
from keras.layers.embeddings import Embedding
from keras.layers.merge import Concatenate
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import json
import pickle
import wget

#train_path = sys.argv[1]
test_path = sys.argv[1]
output_path = sys.argv[2]

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
#sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))



#####################
###   parameter   ###
#####################
split_ratio = 0.1
embedding_dim = 100
nb_epoch = 100
batch_size = 128
max_article_length = 306

################
###   Util   ###
################
def read_data(path,training):
    print ('Reading data from ',path)
    with open(path,'r') as f:
    
        tags = []
        articles = []
        tags_list = []
        
        f.readline()
        for line in f:
            if training :
                start = line.find('\"')
                end = line.find('\"',start+1)
                tag = line[start+1:end].split(' ')
                article = line[end+2:]
                
                for t in tag :
                    if t not in tags_list:
                        tags_list.append(t)
               
                tags.append(tag)
            else:
                start = line.find(',')
                article = line[start+1:]
            
            articles.append(article)
            
        if training :
            assert len(tags_list) == 38,(len(tags_list))
            assert len(tags) == len(articles)
    return (tags,articles,tags_list)

def get_embedding_dict(path):
    embedding_dict = {}
    with open(path,'r') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            coefs = np.asarray(values[1:],dtype='float32')
            embedding_dict[word] = coefs
    return embedding_dict

def get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim):
    embedding_matrix = np.zeros((num_words,embedding_dim))
    for word, i in word_index.items():
        if i < num_words:
            embedding_vector = embedding_dict.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix

def to_multi_categorical(tags,tags_list): 
    tags_num = len(tags)
    tags_class = len(tags_list)
    Y_data = np.zeros((tags_num,tags_class),dtype = 'float32')
    for i in range(tags_num):
        for tag in tags[i] :
            Y_data[i][tags_list.index(tag)]=1
        assert np.sum(Y_data) > 0
    return Y_data

def split_data(X,Y,split_ratio):
    indices = np.arange(X.shape[0])  
    np.random.shuffle(indices) 
    
    X_data = X[indices]
    Y_data = Y[indices]
    
    num_validation_sample = int(split_ratio * X_data.shape[0] )
    
    X_train = X_data[num_validation_sample:]
    Y_train = Y_data[num_validation_sample:]

    X_val = X_data[:num_validation_sample]
    Y_val = Y_data[:num_validation_sample]

    return (X_train,Y_train),(X_val,Y_val)

###########################
###   custom metrices   ###
###########################
def f1_score(y_true,y_pred):
    thresh = 0.4
    y_pred = K.cast(K.greater(y_pred,thresh),dtype='float32')
    tp = K.sum(y_true * y_pred,axis=-1)
    
    precision=tp/(K.sum(y_pred,axis=-1)+K.epsilon())
    recall=tp/(K.sum(y_true,axis=-1)+K.epsilon())
    return K.mean(2*((precision*recall)/(precision+recall+K.epsilon())))

#########################
###   Main function   ###
#########################
def main():
    wget.download("http://www.csie.ntu.edu.tw/~b04902102/MLhw5_model")
    ### read training and testing data
    #(Y_data,X_data,tag_list) = read_data(train_path,True)
    (_, X_test,_) = read_data(test_path,False)
    #all_corpus = X_data + X_test
    #print ('Find %d articles.' %(len(all_corpus)))
    ### tokenizer for all data
    tokenizer = Tokenizer()
    #tokenizer.fit_on_texts(all_corpus)
    with open('word_index', 'rb') as handle:
        tokenizer.word_index = pickle.load(handle)
    with open('label_mapping', 'rb') as handle:
        tag_list = pickle.load(handle)
    word_index = tokenizer.word_index
    ### convert word sequences to index sequence
    #print ('Convert to index sequences.')
    #train_sequences = tokenizer.texts_to_sequences(X_data)
    test_sequences = tokenizer.texts_to_sequences(X_test)
        ### padding to equal length
    #print ('Padding sequences.')
    #train_sequences = pad_sequences(train_sequences)
    #max_article_length = train_sequences.shape[1]
    test_sequences = pad_sequences(test_sequences,maxlen=max_article_length)
    ###
    #train_tag = to_multi_categorical(Y_data,tag_list) 
    ### split data into training set and validation set
    #(X_train,Y_train),(X_val,Y_val) = split_data(train_sequences,train_tag,split_ratio)

    ### get mebedding matrix from glove
    #print ('Get embedding dict from glove.')
    #embedding_dict = get_embedding_dict('data/glove.6B.%dd.txt'%embedding_dim)
    
    #print ('Found %s word vectors.' % len(embedding_dict))
    num_words = len(word_index) + 1
    #print ('Create embedding matrix.')
    #embedding_matrix = get_embedding_matrix(word_index,embedding_dict,num_words,embedding_dim)
    ### build model
    print ('Building model.')
    I = Input((max_article_length, ))
    E = Embedding(input_dim=num_words, output_dim=embedding_dim, trainable=False)(I)
    R = GRU(256, activation='tanh', dropout=0.5, recurrent_dropout=0.3)(E)
    D = Dense(units=256, kernel_initializer="glorot_normal")(R)
    D = Activation('relu')(D)
    D = Dropout(0.25)(D)
    D = Dense(128, kernel_initializer="glorot_normal")(D)
    D = Activation('relu')(D)
    D = Dropout(0.25)(D)
    D = Dense(64, kernel_initializer="glorot_normal")(D)
    D = Activation('relu')(D)
    D = Dropout(0.5)(D)
    O = Dense(units=38, activation='sigmoid', kernel_initializer="glorot_normal")(D)
    model = Model(inputs=I, outputs=O)
    model.summary()

    adam = Adam(lr=0.001, clipvalue=0.5)
    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=[f1_score])
    
    #earlystopping = EarlyStopping(monitor='val_f1_score', patience = 10, verbose=1, mode='max')
    #checkpoint = ModelCheckpoint(filepath='model/best.hdf5',
    #                             verbose=1,
    #                             save_best_only=True,
    #                             save_weights_only=True,
    #                             monitor='val_f1_score',
    #                             mode='max')
    #hist = model.fit(X_train, Y_train, 
    #                 validation_data=(X_val, Y_val),
    #                 epochs=nb_epoch, 
    #                 batch_size=batch_size,
    #                 callbacks=[checkpoint])
    model.load_weights("MLhw5_model")
    #model.load_weights("model/best2.hdf5")
    Y_pred = model.predict(test_sequences)
    thresh = 0.4
    with open(output_path,'w') as output:
        print ('\"id\",\"tags\"',file=output)
        Y_pred_thresh = (Y_pred > thresh).astype('int')
        for index,labels in enumerate(Y_pred_thresh):
            labels = [tag_list[i] for i,value in enumerate(labels) if value==1 ]
            labels_original = ' '.join(labels)
            print ('\"%d\",\"%s\"'%(index,labels_original),file=output)

if __name__=='__main__':
    main()