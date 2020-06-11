# -*- coding: utf-8 -*-
"""
Created on Wed May 16 22:06:18 2020

@author: @hardik
"""

import numpy as np
import re
import string
import nltk
import multiprocessing
import pandas as pd
import time
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.models.word2vec as w2v

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score, plot_confusion_matrix, classification_report
from sklearn.preprocessing import LabelBinarizer

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

'''
remove_punctAndAlphanumeric implemented to remove the punctuations and other alphanumeric characters in the text
Parameters: a string
Returns: a string with the removed elements
''' 
def remove_punctAndAlphanumeric(text):
    text_noExtra = "".join([char for char in text if char not in string.punctuation and (char in string.ascii_letters or char in char in string.whitespace)])
    return text_noExtra

'''
remove_stop implemented to remove the stopwords except "IS" because "IS" is short form for ISIS
Parameters: a list of tokens
Returns: a list of tokens with stopword tokens removed
'''
def remove_stop(tokens):
    tokens_noStop = [word for word in tokens if word not in stopwords]
    return tokens_noStop

'''
Parameters: a list of tokens
Returns: a list of tokens with stemmed words
'''
ps = nltk.PorterStemmer()

def stemming(tokens):
    lemma_tokens = [ps.stem(word) for word in tokens]
    return lemma_tokens

'''
cleanAndTokenize implemented for the cleaning, tokenizing and lemmatizing the tweets
Parameters: a string (tweets)
Returns: a list of tokens (cleaned and stemmed)
'''
def cleanAndTokenize(text):
    cleanedText = remove_punctAndAlphanumeric(text)
    tokens = re.split('\W+',cleanedText)
    
    tokens_removedStop = remove_stop(tokens)
    tokens_stemmed = stemming(tokens_removedStop)
    return tokens_stemmed

def SVM(X_train_vect, y_train, X_test_vect, y_test):
    sv = svm.SVC(C=10, kernel='linear', gamma=0.1, class_weight='balanced')

    start = time.time()
    svm_model = sv.fit(X_train_vect,y_train)
    end = time.time()
    fit_time = round((end - start),3)
    
    start = time.time()
    y_pred=svm_model.predict(X_test_vect)
    end = time.time()
    pred_time = round((end - start),3)
    
    print('Fit Time: {} / Pred Time: {} -------- '.format(fit_time, pred_time))

    print(classification_report(y_test, y_pred, target_names=target_names))
    statistics(y_test, y_pred)
    
def RF(X_train_vect, y_train, X_test_vect, y_test):
    rf = RandomForestClassifier(n_estimators=50, max_depth=25, class_weight='balanced', n_jobs=-1)

    start = time.time()
    rf_model = rf.fit(X_train_vect, y_train)
    end = time.time()
    fit_time = round((end - start),3)
    
    start = time.time()
    y_pred = rf_model.predict(X_test_vect)
    end = time.time()
    pred_time = round((end - start),3)
    
    print('Fit Time: {} / Pred Time: {} -------- '.format(fit_time, pred_time))
    statistics(y_test, y_pred)

def statistics(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(35, 35))
    print(classification_report(y_test, y_pred, target_names=target_names))
    #plt.xticks(rotation=90)
    #plot_confusion_matrix(svm_model, X_test_vect, y_test, ax=ax, values_format='d', xticks_rotation=90, cmap=plt.cm.Blues)
    #plt.show()
    #plt.savefig("RFConfusionMatrix.PNG")
    cm = confusion_matrix(y_test, y_pred)
    #print("Confusion matrix:\n")
    #print(cm)
    lb = LabelBinarizer()
    lb.fit(y_test)
    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)
    auc = roc_auc_score(y_test, y_pred, average='weighted', multi_class='ovr')
    print("\n\nAUC Score:", auc)
   
    FP = cm.sum(axis=0) - np.diag(cm)  
    FN = cm.sum(axis=1) - np.diag(cm)
    TP = np.diag(cm)
    TN = cm.sum() - (FP + FN + TP) 
    
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    print("\nSensitivity of classification: ", TPR.mean())
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    print("\nSpecificity of classification: ", TNR.mean())

def get_vector(sentences):
    # convert to lowercase, ignore all special characters - keep only
    # alpha-numericals and spaces
    #sentence = re.sub(r'[^A-Za-z0-9\s]', r'', str(sentence).lower())
    vectors = [wine2vec.wv[w] for w in sentences
               if w in wine2vec.wv]

    v = np.zeros(wine2vec.vector_size)
    if (len(vectors) > 0):
        v = (np.array([np.sum(x) for x in zip(*vectors)])) / v.size
    return v

#Using maximum 300 characters from every column
pd.set_option('display.max_colwidth', 300)
data = pd.read_csv("C:\\Users\\shobhit\\Downloads\\winemag-data_first150k.csv")
stopwords = nltk.corpus.stopwords.words('english')
labels = data['variety']

varietal_counts = labels.value_counts()
n = 15
mf_labels = data['variety'].value_counts()[:n].index.tolist()
booleans = []

for vari in data.variety:
  if vari in mf_labels:
    booleans.append(True)
  else:
    booleans.append(False)
    
is_frequent_label = pd.Series(booleans)
data = data[is_frequent_label]
data['cleaned_descr'] = data['description'].apply(lambda x: cleanAndTokenize(x.lower()))
cleaned_tokens = data['cleaned_descr'].values.tolist()
num_features = 100
min_word_count = 10
num_workers = multiprocessing.cpu_count()
context_size = 10
downsampling = 1e-3
seed=1993

wine2vec = w2v.Word2Vec(
    sg=1,
    seed=seed,
    workers=num_workers,
    size=num_features,
    min_count=min_word_count,
    window=context_size,
    sample = downsampling
)
wine2vec.build_vocab(cleaned_tokens)
wine2vec.train(cleaned_tokens, total_examples=wine2vec.corpus_count, epochs=5)

data['sentence_vec'] = data['cleaned_descr'].apply(lambda x: get_vector(x))
X_train, X_test, y_train, y_test = train_test_split(data[['sentence_vec']], data['variety'], test_size=0.2)

X_train_vect =  pd.DataFrame(X_train.values.tolist())
X_test_vect = pd.DataFrame(X_test.values.tolist())

target_names = mf_labels
target_names = list(dict.fromkeys(target_names))

RF(X_train_vect, y_train, X_test_vect, y_test)
#SVM(X_train_vect, y_train, X_test_vect, y_test)



