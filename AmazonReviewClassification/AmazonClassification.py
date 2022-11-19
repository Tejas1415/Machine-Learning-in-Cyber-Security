# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 21:27:18 2022

@author: Tejas

Amazon Review Classification
"""
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn import naive_bayes, svm
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

#read the reviews and their polarities from a given file
def loadData(fname):
    reviews=[]
    labels=[]
    f=open(fname)
    for line in f:
        review,rating=line.strip().split('\t') 
        reviews.append(review.lower())    
        labels.append(int(rating))
    f.close()
    return reviews,labels

rev_train,labels_train=loadData('reviews_train.txt')
rev_test,labels_test=loadData('reviews_test.txt')


#Build a counter based on the training dataset
#counter = CountVectorizer(ngram_range=(2,3))
counter = TfidfVectorizer(stop_words="english",sublinear_tf=True)
counter.fit(rev_train)


#count the number of times each term appears in a document and transform each doc into a count vector
counts_train = counter.transform(rev_train)#transform the training data
counts_test = counter.transform(rev_test)#transform the testing data


counts_train_df = pd.DataFrame(counts_train.toarray())
counts_test_df = pd.DataFrame(counts_test.toarray())


clf = SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42,max_iter=5000, tol=None)
clf.fit(counts_train_df,labels_train)
pred = clf.predict(counts_test_df)

print(accuracy_score(pred,labels_test)*100)


clf = RandomForestClassifier().fit(counts_train_df,labels_train)
pred = clf.predict(counts_test_df)

print(accuracy_score(pred,labels_test)*100)








































