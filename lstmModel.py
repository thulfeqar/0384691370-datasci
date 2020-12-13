#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from tensorflow.keras.layers import Embedding,Flatten,Dense,LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

def ltsmModel(data):
    one_hot_rep=[one_hot(words,5000) for words in data.filteredText]
    model=Sequential()
    model.add(Embedding(vocab_size,30,input_length=sent_length))
    model.add(LSTM(100))
    model.add(Dense(1,activation='sigmoid'))
    print(model.summary())
    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    docs=pad_sequences(one_hot_rep,padding='pre',maxlen=250)
    X=np.asarray(docs)
    y=np.asarray(data['label'])
    X = docs
    y = data['label']
    X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=42)
    train_model=model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=5)
    plt.plot(train_model.history['accuracy'],'b',label='train_accuracy')
    plt.plot(train_model.history['val_accuracy'],'r',label='val_accuracy')
    plt.legend()
    plt.show()
    pred=model.predict(X_test)
    
    auc = roc_auc_score(y_test, pred)
    print('LSTM: ROC AUC=%.3f' % (auc))
    fpr, tpr, _ = roc_curve(y_test, pred)
    plt.plot(fpr, tpr, marker='.', label='LSTM')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.show()

