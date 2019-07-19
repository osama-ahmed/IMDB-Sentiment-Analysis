# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 22:35:10 2019

@author: Osama
"""

from keras.models import load_model
from keras import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
import h5py
import pickle

def predict(review):
    max_words=10000
    maxlen=500
    
    with open(r'data/wordIndex.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
        word_index = pickle.load(f)
         
    matrix=np.zeros((1, maxlen))
    i=0
    for word in review.split():
        if i > 499:
            break
        try:
            if word_index[word]<=max_words:
                matrix[0, i]=word_index[word]
                i=i+1
        except:
            continue

    

        
    myModel=load_model(r'model/model.h5')
    #print(myModel.predict(matrix))
    return myModel.predict(matrix)[0]
#    
#result=np.zeros(len(texts))
#i=0
#for text in texts:
#    label=predict(text)
#    if label[0] >= .5:
#        result[i]=1
#    else:
#        result[i]=0
#    i=i+1
#    if (i%10)==0:
#        print(i)
#        
#        
#errors=0
#i=0
#for r in finalResult:
#    if r!=labels[i]:
#        errors=errors+1
#        
#print(errors)
