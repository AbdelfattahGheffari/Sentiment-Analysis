# -*- coding: utf-8 -*-
"""
@author:DataScientist
if you have any question My Email is:
gheffari.abdelfattah@gmail.com 

"""

from sklearn.svm import LinearSVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer ,TfidfTransformer,TfidfVectorizer
#from sklearn.feature_selection import KFold
from sklearn.metrics import accuracy_score , classification_report,confusion_matrix
from sklearn.linear_model import    LogisticRegression 
import matplotlib.pyplot as plt 
import pickle
import re
import pandas as pd
import pretraitement as pt
import joblib
    
Path=r"C:\Users\Data Scientist\Desktop\projet fin d'etude\Source codes\DataSet\tweets"
MyDataSet=load_files(Path,decode_error='ignore' ,shuffle=True,encoding='utf-8')
mydata=MyDataSet.data
stop_words=pt.get_stop_words()    
data_clean=[]

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
    

#Features_Selection

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X=vectorizer.fit_transform(MyDataSet.data)

#Split the DataSet
X_train, X_test,target_train,target_test=train_test_split(X,MyDataSet.target,test_size=0.2 )

#Training

#LinearSVC
LSVC=LinearSVC().fit(X_train, target_train)

#LogisticRegression
LR = LogisticRegression(random_state=0).fit(X_train, target_train)

#k-nearest neighbor
KNN = KNeighborsClassifier()
KNN.fit(X_train, target_train)

#testing our models
print('\n\n********************LSVC CLASSIFIER*************\n\n')   

#lsvc
y_pred_lsvc=LSVC.predict(X_test)
print("\nResults pour  LSVC...")
score_lsvm = accuracy_score(target_test, y_pred_lsvc)
print("accuracy:", score_lsvm*100,"%")
print("Reports:",classification_report(target_test, y_pred_lsvc))
print("confusion_matrix :", confusion_matrix(target_test,y_pred_lsvc))

print('\n\n********************Logistic regression CLASSIFIER*************\n\n')
#LR
y_pred_LR=LR.predict(X_test)
print("\nResults pour LR...")
score_LR=accuracy_score(target_test, y_pred_LR)
print("accuracy:",score_LR*100,"%")
print("Reports:",classification_report(target_test, y_pred_LR))


print('\n\n********************KNN CLASSIFIER*************\n\n')
#KNN
y_pred_KNN=KNN.predict(X_test)
print("\nResults pour KNN...")
score_KNN=accuracy_score(target_test, y_pred_KNN)
print("accuracy:",score_KNN*100,"%")
# print("Reports:",classification_report(target_test, y_pred_KNN))


#save models
vecFileName2 = 'TfIDF_Vectorizer.pk'
with open(vecFileName2, 'wb') as fin1:
         joblib.dump(vectorizer, fin1)
    
    
mdlFileName = 'LR_model2.pk'
with open(mdlFileName, 'wb') as fin:
          joblib.dump(LR, fin)