# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 08:40:40 2019

@author: Osama
"""

from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential, Model
from keras.layers import Dense, Embedding, Bidirectional, CuDNNGRU
from keras.layers import LSTM, Input, GlobalMaxPool1D, Dropout
from keras.datasets import imdb
import os
import numpy as np
import pickle

max_features = 10000
# cut texts after this number of words (among top max_features most common words)
maxlen = 500
batch_size = 32
embed_size=300

print('Loading data...')
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')

print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)


glove_dir = r'C:\Users\Osama\Downloads\DL Workspace\models'

embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.300d.txt'), encoding="utf8")
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        continue
f.close()

print('Found %s word vectors.' % len(embeddings_index))

with open(r'E:\output\My-Projects\Machine learning\IMDB Sentiment Analysis\data\wordIndex.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        word_index = pickle.load(f)
    

embedding_dim = 300

embedding_matrix = np.zeros((max_features, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_features:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector


print('Build model...')
#model = Sequential()
#model.add(Embedding(max_features, 128))
#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2,
#                            return_sequences=True)))
#model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
#model.add(Dense(1, activation='sigmoid'))
inp = Input(shape=(maxlen,))
x = Embedding(max_features, embed_size)(inp)
x = Bidirectional(CuDNNGRU(64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dense(16, activation="relu")(x)
x = Dropout(0.5)(x)
x = Dense(1, activation="sigmoid")(x)
model = Model(inputs=inp, outputs=x)
model.summary()


model.layers[1].set_weights([embedding_matrix])
#model.layers[0].trainable = False

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
history=model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=15,
          validation_split=0.2)
score, acc = model.evaluate(x_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
model.save('model.h5')