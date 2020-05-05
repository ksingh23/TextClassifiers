# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import numpy as np
from collections import defaultdict, Counter
import os.path
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import datetime


def compute_tf_idf(tf, idf):
    '''
    :param tf: the output of the compute_tf function
    :param idf: the output of the compute_idf function
    :return: a dictionary where the keys = words and the values
            = vectors of len(# of training samples) and the tf-idf
            score of each term in that document
    '''
    tf_idf = {term: 0 for term in tf.keys()}
    for term, vec in tf.items():
        tf_idf[term] = [v * idf[term] for v in vec]
    return tf_idf



def compute_total_word_frequencies(dir_path, valid_words):
    '''
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :return: a dictionary where keys = words and values = # of documents in which
            that word appears
    '''
    frequencies = {word: 0 for word in valid_words}
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4]) 
            words = nltk.word_tokenize(content)
            new_words = [word.lower() for word in words if word.isalpha()]
            new_words = [word.lower() for word in new_words if word not in stop_words]
            new_words = set(new_words)
            for word in new_words:
                frequencies[word] += 1
    return frequencies


def compute_tf(dir_path, valid_words):
    '''
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :return: a dictionary where keys = words and values = a vector of the tf score 
            for each document (so the length of each of these vectors is the length
            of the training set)
    '''
    num_docs = len([name for name in os.listdir(dir_path) if os.path.isfile(dir_path + "\\" + name)])
    tf_scores = {word: ([0] * num_docs) for word in valid_words}
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4]) 
            words = nltk.word_tokenize(content)
            new_words = [word.lower() for word in words if word.isalpha()]
            new_words = [word.lower() for word in new_words if word not in stop_words]
            frequencies = Counter(new_words)
            for term in tf_scores.keys():
                tf_scores[term][num-1] = frequencies[term]/len(new_words)
    return tf_scores


def compute_idf(dir_path, valid_words, frequencies):
    '''
    IDF is inverse document frequency, defined as 
    (# of total documents)/(# of occurrences of the word across all documents)
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param frequencies: the return value of the compute_total_word_frequencies
                        function, see above
    :return: a dictionary where the keys = words and values = the IDF score as defined above
    '''
    num_docs = len([name for name in os.listdir(dir_path) if os.path.isfile(dir_path + "\\" + name)])
    idf_scores = {word: 0 for word in valid_words}
    for key in idf_scores.keys():
        idf_scores[key] = np.log(num_docs/(frequencies[key] + 1))
    return idf_scores


def get_valid_words(dir_path, stop_words):
    '''
    Utility function that determines the set of valid words 
    to be used for classification and probability calculation
    :param dir_path: a path to the directory containing 
                    all the training samples
    :param stop_words: a set of words like "the", "and", etc
                        that should be stripped out of any computations
    :return: a Python dictionary where the keys = valid words and the 
            values = True, so we can use "key in dict" for future access
            in guaranteed constant time
    '''
    valid_words = defaultdict(bool)
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            words = nltk.word_tokenize(content)
            new_words = [word.lower() for word in words if word.isalpha()]
            new_words = [word.lower() for word in new_words if word not in stop_words]
            for word in new_words:
                if word not in valid_words:
                    valid_words[word] = True
    return valid_words


if __name__ == '__main__':
    start = float(datetime.datetime.utcnow().timestamp())
    st = LancasterStemmer() 
    words = ['easily', 'easier', 'easy', 'easiest', 'ease', 'eases']
    words = nltk.pos_tag(words)
    print(words)
    stems = [st.stem(word) for word, tag in words]
    print(stems)
    # lem = WordNetLemmatizer()
    # lemmas = [lem.lemmatize(w, get_wordnet_part_of_speech(t.upper())) for w,t in words]
    # print(lemmas)
    end = float(datetime.datetime.utcnow().timestamp())
    print(end - start)    

    stop_words = set(stopwords.words('English'))
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\MiniTrainingSet"
    valid_words = get_valid_words(dir_path, stop_words)
    frequencies = compute_total_word_frequencies(dir_path, valid_words)
    tf = compute_tf(dir_path, valid_words)
    idf = compute_idf(dir_path, valid_words, frequencies)
    tf_idf = compute_tf_idf(tf, idf)
    for key, value in idf.items():
        print(key, value)


