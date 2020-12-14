# -*- coding: utf-8 -*-
#Additional code for accuracy optimization
#Keras tokenizer instead of OHE 
#94 AUC, 87 val & train accuracy after 6 epochs

from tensorflow.keras.layers import Embedding,Flatten,Dense,LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from keras.preprocessing.text import Tokenizer
import numpy as np
from keras.callbacks import EarlyStopping

x = df['text'].values
y = df['label'].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=142)

vocab_size=len(tokenizer.word_index)+1

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(x)
xtrain= tokenizer.texts_to_sequences(x_train)
xtest= tokenizer.texts_to_sequences(x_test) 

maxlen=99
embedding_dim=100

XXtrain=pad_sequences(xtrain,padding='post', maxlen=maxlen)
XXtest=pad_sequences(xtest,padding='post', maxlen=maxlen)

model=Sequential()
model.add(Embedding(input_dim=vocab_size,
      output_dim=embedding_dim))
model.add(LSTM(100))
model.add(Dense(1,activation='sigmoid'))
print(model.summary())
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_model=model.fit(XXtrain,y_train,validation_data=(XXtest,y_test),epochs=25, 
                      callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
plt.plot(train_model.history['accuracy'],'b',label='train_accuracy')
plt.plot(train_model.history['val_accuracy'],'r',label='val_accuracy')
plt.legend()
plt.show()
pred=model.predict(xtest)
    
auc = roc_auc_score(y_test, pred)
print('LSTM: ROC AUC=%.3f' % (auc))
fpr, tpr, _ = roc_curve(y_test, pred)
plt.plot(fpr, tpr, marker='.', label='LSTM')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.show()