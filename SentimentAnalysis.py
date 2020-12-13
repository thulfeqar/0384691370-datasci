#!/usr/bin/env python
# coding: utf-8

# In[1]:


from nltk.sentiment.vader import SentimentIntensityAnalyzer
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

def calculateSentiment(data):
    sid = SentimentIntensityAnalyzer()
    sentimentTextScore = []
    sentimentText = []
    for t in data.filteredText:
        scores = sid.polarity_scores(t)

        if(scores['compound'] < 0):
            sentimentText.append(1) #1 is negative
        else:
            sentimentText.append(0)
        sentimentTextScore.append(scores['compound'])
        
    sentimentTitleScore = []
    sentimentTitle = []
    
    for t in data.filteredTitle:
        scores = sid.polarity_scores(t)
        sentimentTitleScore.append(scores['compound'])
        if(scores['compound'] < 0):
            sentimentTitle.append(1) #1 is negative
        else:
            sentimentTitle.append(0)
            
    data['sentimentTitleScore'] = sentimentTitleScore
    data['sentimentTitle'] = sentimentTitle
    data['sentimentTextScore'] = sentimentTextScore
    data['sentimentText'] = sentimentText
    
    
    return(data)


def plotSentimentBar(data,column,metric):
    if (metric == 'sum'):
        df = data.groupby(['label'])[column].sum()
        df.plot.bar()
    else:
        df = data.groupby(['label'])[column].mean()
        df.plot.bar()

def plotGraph(df_sentiment):
    plotSentimentBar(df_sentiment,'sentimentText','sum')
    plotSentimentBar(df_sentiment,'sentimentTextScore','mean')
    plotSentimentBar(df_sentiment,'sentimentTitle','sum')
    plotSentimentBar(df_sentiment,'sentimentTitleScore','mean')
    
def saveToPickleSentiment(df):
    filename_sentiment = 'df_clean_sentiment' #C:\Users\13236\Downloads
    outfile = open(filename_sentiment,'wb')
    pickle.dump(df,outfile)
    outfile.close()

def loadPicklefile():
    filename = 'df_clean'
    infile = open(filename,'rb')
    df_clean = pickle.load(infile, encoding='bytes')
    return df_clean

def mainSentimentAnalysis():
    df_clean = loadPicklefile()
    df_sentiment = calculateSentiment(df_clean)
    saveToPickleSentiment(df_sentiment)
    plotGraph(df_sentiment)


if __name__ == "__main__":
    mainSentimentAnalysis()



