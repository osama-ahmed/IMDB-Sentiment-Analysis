# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 19:30:19 2019

@author: Osama
"""

import matplotlib.pyplot as plt
import random
import os
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

def plotResults(history):
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(acc))
    
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    
    plt.show()
    
    
def preprocessImdb(dataDirectory,
                   maxWords,
                   maxLength,
                   IsValidationDataNeeded=False, 
                   trainingSamplesNo=0, 
                   ValidationSamplesNo=0):
    
    labels = []
    texts = []
    
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(dataDirectory, label_type)
        for fname in os.listdir(dir_name):
            try:
                if fname[-4:] == '.txt':
                    f = open(os.path.join(dir_name, fname), encoding="utf8")
                    texts.append(f.read())
                    f.close()
                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)
            except:
                continue
    
    
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    
    texts[:], labels[:] = zip(*combined)
    
    tokenizer = Tokenizer(num_words=maxWords)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    
    word_index = tokenizer.word_index
    
    data = pad_sequences(sequences, maxlen=maxLength)

    labels = np.asarray(labels)
    
#    indices = np.arange(data.shape[0])
#    np.random.shuffle(indices)
#    data = data[indices]
#    labels = labels[indices]
    
    if IsValidationDataNeeded==True:
        trainingData = data[:trainingSamplesNo]
        trainingLabels = labels[:trainingSamplesNo]
        validationData = data[trainingSamplesNo: trainingSamplesNo + ValidationSamplesNo]
        validationLabels = labels[trainingSamplesNo: trainingSamplesNo + ValidationSamplesNo]

        return word_index, trainingData, trainingLabels, validationData, validationLabels
    else:
        return word_index, data, labels
    