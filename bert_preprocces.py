# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 07:47:09 2019

@author: Osama
"""

import os
import numpy as np


def createLists(imdbDir):
    labels = []
    texts = []
    
    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(imdbDir, label_type)
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
    return texts, labels

def shuffleLists(texts, labels):
    import random
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    
    texts[:], labels[:] = zip(*combined)
    return texts, labels
    
            
def createCSV(texts, labels, outFile):
    import csv
    i=0
    with open(outFile,'w', newline='', encoding='utf-8') as file:
        for text,label in zip(texts,labels):
            try:
                mylist=[]
                mylist.append(str(i))
                mylist.append(str(label))
                mylist.append('a')
                mylist.append(text)
                
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(mylist)
                i=i+1
                #print(text)
            except:
                continue
            
def createCSVForTestSet(texts, labels, outFile):
    import csv
    i=0
    with open(outFile,'w', newline='', encoding='utf-8') as file:
        header=[]
        header.append('id')
        header.append('sentence')
        wr = csv.writer(file, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        
        for text,label in zip(texts,labels):
            try:
                mylist=[]
                mylist.append(str(i))
                mylist.append(text)
                
                wr = csv.writer(file, quoting=csv.QUOTE_ALL)
                wr.writerow(mylist)
                i=i+1
                #print(text)
            except:
                continue


#loading data and preproccing
imdb_dir = r'C:\Users\Osama\Downloads\DL Workspace\Data\aclImdb'
train_dir = os.path.join(imdb_dir, 'train')
test_dir = os.path.join(imdb_dir, 'test')

#train and validation set
texts, labels = createLists(train_dir)
texts, labels=shuffleLists(texts, labels)
createCSV(texts[:20000], labels[:20000], 'train.csv')
createCSV(texts[20000:25000], labels[20000:25000], 'dev.csv')

#test set
texts, labels = createLists(test_dir)
#texts, labels=shuffleLists(texts, labels)
createCSVForTestSet(texts, labels, 'test.csv')



import pandas as pd
path='E:/output/My-Projects/Machine learning/IMDB Sentiment Analysis/train.csv'
df = pd.read_csv(path, 'utf-8')
path='E:/output/My-Projects/Machine learning/IMDB Sentiment Analysis/train.tsv'
df.to_csv(path, sep='\t', index=False, header=False)
# if you are creating test.tsv, set header=True instead of False

path='E:/output/My-Projects/Machine learning/IMDB Sentiment Analysis/dev.csv'
df = pd.read_csv(path, 'utf-8')
path='E:/output/My-Projects/Machine learning/IMDB Sentiment Analysis/dev.tsv'
df.to_csv(path, sep='\t', index=False, header=False)

path='E:/output/My-Projects/Machine learning/IMDB Sentiment Analysis/test.csv'
df = pd.read_csv(path, 'utf-8')
path='E:/output/My-Projects/Machine learning/IMDB Sentiment Analysis/test.tsv'
df.to_csv(path, sep='\t', index=False, header=True)
