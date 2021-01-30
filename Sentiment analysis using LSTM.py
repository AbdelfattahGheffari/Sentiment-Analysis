# -*- coding: utf-8 -*-
"""
@author:DataScientist
if you have any question My Email is:
gheffari.abdelfattah@gmail.com 

"""

from keras.preprocessing import  sequence
from keras.models import Sequential
from keras.layers import Dense ,Dropout , Embedding , LSTM, Bidirectional,SpatialDropout1D
import codecs
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import re 
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt 
import pretraitement as pt
import report_metrics as report 

#Path=r"Path of DataSet"
#Load the DataSet
MyDataSet=load_files(Path,decode_error='ignore' ,shuffle=True,encoding='utf-8')
mydata=MyDataSet.data
stop_words=pt.get_stop_words()    
data_clean=[]
#Cleaning the Data
for tweet in mydata:
    tweet=pt.pretraitement(tweet)
    tweet_split= tweet.split()
    tweet=[] 
    for word in tweet_split:
          word=word.replace(u'\ufeff', '')
          tweet.append(word)
    data_clean.append(tweet)
    
for i in range (len(data_clean)):
    MyDataSet.data[i]=data_clean[i]
    MyDataSet.data[i]=' '.join(MyDataSet.data[i]) 


#Hyper-Parameters
max_features =5000
num_classes=1
max_length=100
batch_size=64
embidding_dim=16 #choose number multiple of 2 like The machine (2^N)32 or 64 or 128 .....
dropout_rate=0.5 #Dropout_Layer
hidden_layer_size=256
num_epochs=50


#Tokenization
tok = Tokenizer(num_words=max_features )
tok.fit_on_texts(MyDataSet.data)

#make Sequence for each text
sequences = tok.texts_to_sequences(MyDataSet.data)
sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_length)

#number of words 
vocab_size=len(tok.index_word)+1


#Design Neural Network Architecture with LSTM 

LSTM_model = Sequential()

LSTM_model.add(Embedding(vocab_size, embidding_dim,input_length =max_length))

LSTM_model.add(Dropout(dropout_rate))

LSTM_model.add(LSTM(32,activation='tanh'))

LSTM_model.add(Dropout(dropout_rate))

LSTM_model.add(Dense(num_classes,activation='sigmoid'))

LSTM_model.compile(loss = 'binary_crossentropy', optimizer='adam',metrics=['accuracy',report.f1_m,report.recall_m,report.precision_m])

print(LSTM_model.summary())

#Train Test
X_train, X_test,target_train,target_test=train_test_split(sequences_matrix,MyDataSet.target,test_size=0.2)

#Train Validation
X_train,X_val, target_train ,target_val=train_test_split(X_train,target_train,test_size=0.2)

#EarlyStopping Regularization
es= EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3,min_delta=0.0001)

#history
history=LSTM_model.fit(X_train,  target_train, epochs =1, batch_size=batch_size,validation_data=(X_val,target_val),callbacks=[es])


loss, acc ,F1,recall,precision = LSTM_model.evaluate(X_test, target_test, batch_size=batch_size)



    
print('Test accuracy:', acc) 
print('Test score:', loss)
print('Test F1:', F1)
print('Test recall:', recall)
print('Test precision:', precision)

#"Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#"Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

#We Can ploting F1 & recall & precision