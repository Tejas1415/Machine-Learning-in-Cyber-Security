# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 21:30:33 2022

@author: Tejas
"""

import csv
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
import numpy as np


#function that loads a lexicon of positive words to a set and returns the set
def loadLexicon(fname):
    newLex=set()
    lex_conn=open(fname)
    
    #add every word in the file to the set
    for line in lex_conn:
        newLex.add(line.strip())# remember to strip to remove the lin-change character
    lex_conn.close()

    return newLex


##### 


def parse(input_path, index1, index2):
    import pandas as pd
    
    df = pd.read_csv(input_path, header = None)
    
    review1 = df.iloc[index1, 0]
    review2 = df.iloc[index2, 0]
    
    #load the positive and negative lexicons into sets
    posLex=loadLexicon('positive-words.txt')
    negLex=loadLexicon('negative-words.txt')
    
    
        
    sentences=sent_tokenize(review1) # split the review into sentences 
    
    df_list = []
    for sentence in sentences:
        
        words=word_tokenize(sentence) # split the review into words
    
        tagged_words=nltk.pos_tag(words) # POS tagging for the words in the sentence
    
        
        nouns = []
        #posCount = []
        #negCount = []
        pos = 0
        neg = 0
        for tagged_word in tagged_words:
            if tagged_word[1].startswith('NN'):
                if len(tagged_word[0]) >3:
                    nouns.append(tagged_word[0])
                
            if tagged_word[1].startswith('JJ'):
                if tagged_word[0] in posLex:
                    pos = pos+1
                if tagged_word[0] in negLex:
                    neg = neg+1
        
        df1 = pd.DataFrame()
        df1['nouns'] = nouns
        df1['posCount'] = pos
        df1['negCount'] = neg
        
        df_list.append(df1)
    
    
    df_result1 = pd.concat(df_list, 0)
    
    
    df_result1_ = df_result1.groupby(['nouns']).sum()
    
    df_result1_['sentiment'] = np.where(df_result1_['posCount'] > df_result1_['negCount'], 'Positive', 'Negative')
    df_result1_['sentiment'] = np.where(df_result1_['posCount'] == df_result1_['negCount'], 'Neutral', df_result1_['sentiment'])
    
    
    
    
    
    
    sentences2=sent_tokenize(review2) # split the review into sentences 
    
    df_list2 = []
    for sentence in sentences2:
        
        words=word_tokenize(sentence) # split the review into words
    
        tagged_words=nltk.pos_tag(words) # POS tagging for the words in the sentence
    
        
        nouns = []
        #posCount = []
        #negCount = []
        pos = 0
        neg = 0
        for tagged_word in tagged_words:
            if tagged_word[1].startswith('NN'):
                if len(tagged_word[0]) >3:
                    nouns.append(tagged_word[0])
                
            if tagged_word[1].startswith('JJ'):
                if tagged_word[0] in posLex:
                    pos = pos+1
                if tagged_word[0] in negLex:
                    neg = neg+1
        
        df2 = pd.DataFrame()
        df2['nouns'] = nouns
        df2['posCount'] = pos
        df2['negCount'] = neg
        
        df_list2.append(df2)
    
    
    df_result2 = pd.concat(df_list2, 0)
    
    
    df_result2_ = df_result2.groupby(['nouns']).sum()
    
    df_result2_['sentiment'] = np.where(df_result2_['posCount'] > df_result2_['negCount'], 'Positive', 'Negative')
    df_result2_['sentiment'] = np.where(df_result2_['posCount'] == df_result2_['negCount'], 'Neutral', df_result2_['sentiment'])
    
    
    
    df_result1_ = df_result1_.reset_index()
    df_result2_ = df_result2_.reset_index()
    
    
    opp_opinion = []
    
    df_result1_1 = df_result1_[df_result1_['sentiment'] != 'Neutral']
    df_result2_2 = df_result2_[df_result2_['sentiment'] != 'Neutral']
    
    
    for i in range(0, df_result1_1.shape[0]):
        current_noun = df_result1_1['nouns'].iloc[i]
        sentiment1 = df_result1_1['sentiment'].iloc[i]
        
    
    
        if current_noun in df_result2_2['nouns'].tolist():
            temp = df_result2_2[df_result2_2['nouns'] == current_noun]
            sentiment2 = temp['sentiment'].iloc[0]
            
            
            if sentiment2 != sentiment1:
                opp_opinion.append(current_noun)
    
    
    
    return opp_opinion
    
    

input_path = "C:/Users/Tejas/Downloads/Github/Manasa Assignments/Machine-Learning-in-Cyber-Security/amazonreviews.csv"
index1 = 3
index2 = 4



parse(input_path, 3, 4)






input_file = 'amazonreviews.csv'

#load the positive and negative lexicons into sets
posLex=loadLexicon('positive-words.txt')
negLex=loadLexicon('negative-words.txt')

noun_sentiment={}#maps each noun to the number of times it appears in the same sentence as a positive or negative word
    
fin=open(input_file,encoding='utf8')

reader=csv.reader(fin)

for line in reader: # for each review

    text,rating=line # get the text and rating

    sentences=sent_tokenize(text) # split the review into sentences

    for sentence in sentences: # for each sentence
    
        words=word_tokenize(sentence) # split the review into words
    
        tagged_words=nltk.pos_tag(words) # POS tagging for the words in the sentence

        nouns_in_sentence=set() # set of all the nouns in the sentence
    
        sentiment_word_count=0 # number of positive or negative words in the sentence
    
        #https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html

        for tagged_word in tagged_words:
        
            if tagged_word[1].startswith('NN'): # if the word is a noun

                noun=tagged_word[0].lower() # lower case the noun
                
                if len(noun)<3:continue # ignore nouns with less than 3 characters
                            
                nouns_in_sentence.add(noun) # add the noun to the set
            
            if tagged_word[1].startswith('JJ') and (tagged_word[0] in posLex or tagged_word[0] in negLex): 
                sentiment_word_count+=1
              
                
                
            prev_word=tagged_word
    
        for noun in nouns_in_sentence: # for each noun that we found in the sentence
            noun_sentiment[noun]=noun_sentiment.get(noun,0)+sentiment_word_count

fin.close()