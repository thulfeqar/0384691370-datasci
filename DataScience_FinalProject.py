#!/usr/bin/env python
# coding: utf-8

# In[13]:


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
import SentimentAnalysis
import WordPlot
#nltk.download()
#pip intall 


# In[14]:


#Loads data file from a given path into a dataframe
# ID = Record ID
# Title = Title of the article
# Author = Autor of the Article. Could be an individual, organization or a group of individuals
# Text = The actual article text
# Label = Classifies an article as reliable vs unreliabe (0: reliable and 1: unreliable)
def loadFile(path):
  names = ["id","title","author","text","label"]
  df = pd.read_csv(path,sep = ",",names= names,header = 0)
  df.dropna(how='any', inplace=True) #Drop all rows where any column has NAN
  return df


# In[15]:


#cleans the dataframe to remove special characters, stopwords, spaces, numbers and performs lemmatization and Stemming
#input is a raw data frame and output is a data frame with two additional columns - filteredText and filteredTitle
def cleanFile(data):
    
  lem = WordNetLemmatizer()
  porter=PorterStemmer()

  #Drop NA
  data.dropna(how='any')

  #Remove stop words, sppecial characters and numbers from text
  stop_words = set(stopwords.words('english'))  
  filtered_sentence = []
  filtered_title = []
  for t in data.text: 
    #remove special characters and numbers
    t = re.sub(r"\W+|_", " ", t)
    t = re.sub(r'[0-9]+', '', t)
    #Convert to lower case and tokenize
    word_tokens = word_tokenize(t.lower())
    #remove stop words, Lemmatize and Stem
    filtered_sentence.append(' '.join([lem.lemmatize(porter.stem(w)) for w in word_tokens if not w in stop_words]))
 
  for t in data.title: 
    #remove special characters and numbers
    t = re.sub(r"\W+|_", " ", t)
    t = re.sub(r'[0-9]+', '', t)
    #Convert to lower case and tokenize
    word_tokens = word_tokenize(t.lower())
    #remove stop words, Lemmatize and Stem
    filtered_title.append(' '.join([lem.lemmatize(porter.stem(w)) for w in word_tokens if not w in stop_words]))

  data['filteredText'] = filtered_sentence
  data['filteredTitle'] = filtered_title
  #Returns dataframe with two additional columns - filteredText and filteredTitle
  return data


# In[16]:


#Calculates the scroe for each author based number of unreliable articles vs total number of articles
#Output is a data frame with additional column for author score
def authorScore(data):
    auth = pd.DataFrame(data.groupby('author').apply(lambda x: x['label'].sum()/x['label'].count()))
    auth.columns = ['authorScore']
    data = pd.merge(left=data, right=auth, left_on='author', right_on='author')
    return data
        


# In[17]:


#Performs all the preprossing and stores the output into a pickle file df_clean
#Runs additional plots for unigrams and ngrams and also calculates sentiment score
def mainPreProcessing():
    df = loadFile("C:/Users/13236/Downloads/Fake-News-Dataset-master/fake-news/train.csv") #Enter file Path
    df_clean = cleanFile(df)
    df_clean_auth = authorScore(df_clean)
    filename = 'df_clean' #C:\Users\13236\Downloads
    outfile = open(filename,'wb')
    pickle.dump(df_clean_auth,outfile)
    outfile.close()
    WordPlot.mainWordPlot()
    SentimentAnalysis.mainSentimentAnalysis()


# In[ ]:


if __name__ == "__main__":
  mainPreProcessing()
  #df = loadFile("C:/Users/13236/Downloads/Fake-News-Dataset-master/fake-news/train.csv") #Enter file path
  #df_clean = cleanFile(df)


# In[ ]:




