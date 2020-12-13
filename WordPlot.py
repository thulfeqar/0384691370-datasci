#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize  
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt; plt.rcdefaults()
import numpy as np
import pickle
#nltk.download()
#pip intall 

def frequentWords(wordList,count):
    #corpus = df_clean['filteredText'].values.astype('U')
    cntVector = CountVectorizer().fit(wordList)
    bagOfWords = cntVector.transform(wordList)
    sumWords = bagOfWords.sum(axis=0)
    words_freq = [(word, sumWords[0, idx]) for word, idx in cntVector.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:count],columns=['text','count'])

def frequentNWords(wordList, count):
    cntVector = CountVectorizer(ngram_range=(2, 2)).fit(wordList)
    bagOfWords = cntVector.transform(wordList)
    sumWords = bagOfWords.sum(axis=0)
    words_freq = [(word, sumWords[0, idx]) for word, idx in cntVector.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return pd.DataFrame(words_freq[:count],columns=['text','count'])

def plotGraph(data,title):
    objects = (data['text'])
    y_pos = np.arange(len(objects))
    count = data['count']
    plt.figure(figsize=(10, 5))
    plt.bar(y_pos, count, align='edge', alpha=0.5, width=0.3)
    plt.xticks(y_pos, objects)
    plt.ylabel('Count')
    plt.title(title)
    plt.show()
    
def mainWordPlot():
    filename = 'df_clean'
    infile = open(filename,'rb')
    df_clean = pickle.load(infile, encoding='bytes')
    df_fake = df_clean[df_clean['label'] == 1]
    df_real = df_clean[df_clean['label'] == 0]  
    df_freqWords = frequentWords(df_fake['filteredText'].values.astype('U'), 20)
    df_freqNWords = frequentNWords(df_fake['filteredText'].values.astype('U'), 20)
    plotGraph(df_freqWords,'TextFakeUnigram')
    plotGraph(df_freqNWords,'TextFakeBigram')
    
    df_freqWordsReal = frequentWords(df_real['filteredText'].values.astype('U'), 20)
    df_freqNWordsReal = frequentNWords(df_real['filteredText'].values.astype('U'), 20)
    plotGraph(df_freqWordsReal,'TextRealUnigram')
    plotGraph(df_freqNWordsReal,'TextRealBigram')
    
    df_freqWords = frequentWords(df_fake['filteredTitle'].values.astype('U'), 20)
    df_freqNWords = frequentNWords(df_fake['filteredTitle'].values.astype('U'), 20)
    plotGraph(df_freqWords,'TitleFakeUnigram')
    plotGraph(df_freqNWords,'TitleFakeBigram')
    
    df_freqWordsReal = frequentWords(df_real['filteredTitle'].values.astype('U'), 20)
    df_freqNWordsReal = frequentNWords(df_real['filteredTitle'].values.astype('U'), 20)
    plotGraph(df_freqWordsReal,'TitleRealUnigram')
    plotGraph(df_freqNWordsReal,'TitleRealBigram')


if __name__ == "__main__":
    mainWordPlot()

