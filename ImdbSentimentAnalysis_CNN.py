import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, LSTM, Dropout
from keras import layers
import pickle
from myUtility import preprocessImdb


maxlen = 500  # We will cut reviews after 100 words
training_samples = 20000  # We will be training on 200 samples
validation_samples = 5000  # We will be validating on 10000 samples
max_words =10000  # We will only consider the top 10,000 words in the dataset


#loading data and preproccing
imdb_dir = r'C:\Users\Osama\Downloads\DL Workspace\Data\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')

word_index, x_train, y_train, x_val, y_val= preprocessImdb(train_dir, max_words, maxlen, IsValidationDataNeeded=True, trainingSamplesNo=training_samples, ValidationSamplesNo=validation_samples)  


#import csv
#i=0
#with open('train.csv','w') as file:
#    for text,label in zip(texts,labels):
#        try:
#            mylist=[]
#            mylist.append(str(i))
#            mylist.append(str(label))
#            mylist.append('a')
#            mylist.append(text)
#            
#            wr = csv.writer(file, quoting=csv.QUOTE_ALL)
#            wr.writerow(mylist)
#            i=i+1
#            #print(text)
#        except:
#            continue
#    
                

"""
import numpy as np
np.savetxt('x_train.txt', x_train)
np.savetxt('y_train.txt', y_train)
np.savetxt('x_val.txt', x_val)
np.savetxt('y_val.txt', y_val)


import numpy as np

x_train=np.loadtxt('/content/drive/My Drive/app/x_train.txt')
y_train=np.loadtxt('/content/drive/My Drive/app/y_train.txt')
x_val=np.loadtxt('/content/drive/My Drive/app/x_val.txt')
y_val=np.loadtxt('/content/drive/My Drive/app/y_val.txt')
"""


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

embedding_dim = 300

embedding_matrix = np.zeros((max_words, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector




model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=maxlen))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPooling1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPooling1D())
#model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
#model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.layers[0].set_weights([embedding_matrix])
#model.layers[0].trainable = False

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=4,
                    batch_size=128,
                    validation_data=(x_val, y_val))
model.save_weights('modelWeights.h5')
model.save('model.h5')

from myUtility import plotResults
plotResults(history)


test_dir = os.path.join(imdb_dir, 'test')

word_index, x_test, y_test= preprocessImdb(test_dir, max_words, maxlen)  

#model.load_weights('modelWeights.h5')
model.evaluate(x_test, y_test)



#from keras.models import load_model
#myModel=load_model('preTrainedModel.h5')
#myModel.evaluate(x_test, y_test)

