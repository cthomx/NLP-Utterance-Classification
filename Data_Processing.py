# Importing the required packages
import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords 
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from itertools import compress
import collections
import warnings
warnings.filterwarnings('ignore')

# Importing the datasets
def importData():
    train = pd.read_csv('train.csv', error_bad_lines=False, warn_bad_lines=False)
    test = pd.read_csv('test.csv', error_bad_lines=False, warn_bad_lines=False)
    return train, test

def filterData(train, test):
    # select "utterance" and "context" as X and y
    # only select {'sad', 'jealous', 'joyful', 'terrified'} categories
    train = train.loc[train['context'].isin(['sad', 'jealous', 'joyful', 'terrified'])]
    test = test.loc[test['context'].isin(['sad', 'jealous', 'joyful', 'terrified'])]
    X_train = train[['utterance', 'context']]
    X_test = test[['utterance', 'context']]
    return train, test, X_train, X_test

# Getting the train labels
def initalizeLabels(train, test):
    train_labels_unique = list(train['context'].unique())
    label_mapper = {}
    num = 0
    for label in train_labels_unique:
        label_mapper[label] = num
        num += 1

    train_labels = list(train['context'])
    labels_train_encoded = []
    for label in train_labels:
        labels_train_encoded.append(label_mapper[label])
        
    # Getting test labels
    labels_test = list(test['context'])
    labels_encoded_test = []
    for label in labels_test:
        labels_encoded_test.append(label_mapper[label])
    labels_encoded_test = np.array(labels_encoded_test)
    # note train and test labels are in train_labels_encoded and labels_encoded_test
    return labels_train_encoded, labels_encoded_test

# data preprocessing, remove punctuations from the sentence
# DO NOT REMOVE STOPWORDS here
# https://medium.com/@arunm8489/getting-started-with-natural-language-processing-6e593e349675
# https://machinelearningmastery.com/clean-text-machine-learning-python/
# utilizing methods from the medium / ml mastery article to PreProcess the data
def dataPreProcess(dataFrameObject, targetColumn):
    # set up dataset list
    dataset = list(dataFrameObject[targetColumn])
    
    # set up lemmatizer
    from nltk.stem.wordnet import WordNetLemmatizer 
    lem = WordNetLemmatizer()
    
    # set up contraction fixer 
    import contractions
    
    # clean up text with regex
    for i in range(len(dataset)):
        
        # fix contractions
        dataset[i] = contractions.fix(dataset[i])
        
        # regular expressions 
        dataset[i] = dataset[i].lower()
        dataset[i] = re.sub(r'\W',' ',dataset[i])  # will remove non-word charecters like #,*,% etc
        dataset[i] = re.sub(r'_',' ',dataset[i])   # remove underscores
        dataset[i] = re.sub(r'\d',' ',dataset[i])  # will remove digits
        dataset[i] = re.sub(r'\s+',' ',dataset[i]) # will remove extra spaces
        
        # lemmatize the dataset for verbs
        words = nltk.word_tokenize(dataset[i])
        words = [lem.lemmatize(word,pos='v') for word in words]
        dataset[i] = ' '.join(words)
        
        # lemmatize the dataset for nouns
        words = nltk.word_tokenize(dataset[i])
        words = [lem.lemmatize(word,pos='n') for word in words]
        dataset[i] = ' '.join(words)
        
        # lemmatize the dataset for adjectives
        words = nltk.word_tokenize(dataset[i])
        words = [lem.lemmatize(word,pos='a') for word in words]
        dataset[i] = ' '.join(words)
        
        # lemmatize the dataset for adverbs
        words = nltk.word_tokenize(dataset[i])
        words = [lem.lemmatize(word,pos='r') for word in words]
        dataset[i] = ' '.join(words)
        
        # lemmatize the dataset for satellite adjectives
        words = nltk.word_tokenize(dataset[i])
        words = [lem.lemmatize(word,pos='s') for word in words]
        dataset[i] = ' '.join(words)
    
    return dataset

# clean up stopwords
# source: https://arunm8489.medium.com/getting-started-with-natural-language-processing-6e593e349675
def cleanStopWords(dataset):
    for i in range(len(dataset)):
        # Getting the list of stopwords and appending additional words to it
        stopwords_list = list(set(stopwords.words('english')))
        stopwords_list.extend(['comma', '']) 

        # tokenize dataset
        words = nltk.word_tokenize(dataset[i])
        new = []

        for word in words:
            # Also performed a run with words less than 3 in length removed but it hurt performance, leaving these in
            # if ((word not in stopwords_list) & (len(word) > 3)):
            if word not in stopwords_list:
                new.append(word)
            dataset[i] = ' '.join(new)
    return dataset

    
def vectorizeText(trainDataCleanedWithoutStopWords, testDataCleanedWithoutStopWords):
    # vectorize the text
    count_vectorizer = CountVectorizer()
    X_train = count_vectorizer.fit_transform(traindataCleanedWithoutStopWords)
    X_test = count_vectorizer.transform(testDataCleanedWithoutStopWords)

    # utilize tfidi to 
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True)
    train_embedding_tfidf_transformer = tfidf_transformer.fit_transform(X_train)
    test_embedding_tfidif_transofmrer = tfidf_transformer.transform(X_test)
    return train_embedding_tfidf_transformer, test_embedding_tfidif_transofmrer

# main function if executing this module 
if __name__ == '__main__': 
    train, test = importData() # import raw data
    
    train, test, X_train, X_test= filterData(train, test) # filter data for certain text categories

    labels_train_encoded, labels_encoded_test = initalizeLabels(train, test) # encode labels

    train_data_list_cleaned = dataPreProcess(X_train, 'utterance') # clean dataset with regex and other manipulation like fixing contractions
    test_data_list_cleaned = dataPreProcess(X_test, 'utterance') # clean dataset with regex and other manipulation like fixing contractions

    train_data_stop_removed = cleanStopWords(train_data_list_cleaned) # remove the tokens in the stopwords list from utterance
    test_data_stop_removed = cleanStopWords(test_data_list_cleaned) # remove the tokens in the stopwords list from utterance