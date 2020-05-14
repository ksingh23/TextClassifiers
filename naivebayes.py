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
from operator import itemgetter


def naive_bayes(tf_idf, vectorized_text, prior_probs):
    '''
    :param tf_idf: dictionary where keys = labels and values = dictionary where
                    keys = words and values = tf_idf scores
    :param vectorized_text: words from text that are in valid_words
    :param prior_probs: dictionary where keys = labels and values = the probability
                        of seeing that label in the dataset
    '''
    labels = []
    for label in tf_idf.keys():
        prob = np.log(prior_probs[label])
        conditional = 0.0
        for word in vectorized_text:
            conditional += np.log(tf_idf[label][word])
        prob += conditional
        labels.append((label, prob))
    return sorted(labels, key=itemgetter(1), reverse=True)


def bayes_accuracy_model(num, number_labels, labels, num_samples_desired):
    '''
    :param num: the number of the document being checked, so we can check
                the correct labels for it
    :param number_labels: dictionary where keys = number of sample and
                            values = the set of labels associated with
                            that sample
    :param labels: the set of labels computed by Naive Bayes
    :param num_samples_desired: determines the length of the subset of 
                                the computed labels desired
    '''
    sample_labels = number_labels[num]
    successes = 0
    computed_labels = [x for x,y in labels]
    computed_labels = computed_labels[:num_samples_desired]
    if all(x in computed_labels for x in sample_labels):
        successes += 1
    else:
        diff = set(sample_labels).difference(set(computed_labels))
        if len(diff) < len(computed_labels):
            successes += (len(diff)/len(computed_labels))
    return successes


def vectorize_text(stop_words, valid_words, st, filepath):
    '''
    This function removes non valid words from the text to put it into
    the Naive Bayes classifier
    :param stop_words: a set of words like "the", "and", etc
                        that should be stripped out of any computations
    :param valid_words: dictionary where keys = valid words in the corpus
    :param st: Lancaster Stemmer object
    :param filepath: path to the text file
    :return: a vector of text stripped of stop words and non-valid words
    '''
    with open(filepath, "r") as f:
        content = f.read()
        labels = number_labels[num]
        words = nltk.word_tokenize(content)
        words = [st.stem(word) for word in words]
        new_words = [word.lower() for word in words if word not in stop_words]
        new_words = [word.lower() for word in new_words if word in valid_words.keys()]
    return new_words


def compute_tf_idf(tf, idf):
    '''
    Perhaps I need to restructure this; so that the keys are the document # and the values
    are dictionaries where keys = words and values = tf_idf score for that word
    Then,  computing the "average" tf_idf score becomes a matter of grabbing the articles 
    with a specific label and computing the average of all the values for each key
    
    
    :param tf: the output of the compute_tf function
    :param idf: the output of the compute_idf function
    :return: a dictionary where the keys = document # and the values
            = dictionary where keys = word and value = tf_idf score of 
            that word in that document
    '''
    tf_idf = {num: {word: 0.0 for word in idf.keys()} for num in tf.keys()}
    for num, value in tf.items():
        for word, v in value.items():
            tf_idf[num][word] = tf[num][word] * idf[word]
    return tf_idf
    


def cosine_similarity(avg_tf_idf, tf_idf_vector):
    '''
    This function takes the average TF-IDF vector for 
    every unique label and computes the cosine similarity between in
    and the tf-idf vector for a given sample. 
    :param avg_tf_idf: dictionary where keys = labels and values = dictionary
                        where keys = words and values = the average tf-idf score
                        for that term in documents with that specific label
    :param tf_idf_vector: numpy array 
    '''
    labels = []
    for label in avg_tf_idf.keys():
        # Cosine similarity = (a * b)/(|a| * |b|)
        # Higher cosine similarity = more similar documents
        vector = np.asarray(list(avg_tf_idf[label].values()))
        similarity = np.dot(vector, tf_idf_vector)
        mag_a = np.sqrt(np.dot(vector, vector))
        mag_b = np.sqrt(np.dot(tf_idf_vector, tf_idf_vector))
        denom = np.dot(mag_a, mag_b)
        similarity /= denom
        labels.append((label, similarity))
    return sorted(labels, key=itemgetter(1), reverse=True)


def compute_total_word_frequencies(dir_path, valid_words, st):
    '''
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param st: Lancaster Stemmer object 
    :return: a dictionary where keys = words and values = # of documents in which
            that word appears 
    '''
    frequencies = {word: 0 for word in valid_words}
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4]) 
            words = nltk.word_tokenize(content)
            words = [st.stem(word) for word in words]
            new_words = [word.lower() for word in words if word not in stop_words]
            new_words = [word.lower() for word in new_words if word in valid_words.keys()]
            new_words = set(new_words)
            for word in new_words:
                frequencies[word] += 1
    return frequencies


def compute_tf_new(dir_path, valid_words, st, stop_words):
    '''
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param st: Lancaster stemmer object
    :param stop_words: set of words like "the" and "and" to remove from computations
    :return: a dictionary where keys = document # and values = a dictionary where 
                keys = the word and values = the tf score for that word
            
    '''
    tf_scores = defaultdict(dict)
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4])
            words = nltk.word_tokenize(content)
            words = [st.stem(word) for word in words]
            new_words = [word.lower() for word in words if word not in stop_words]
            # I strip out stop words again because sometimes the stems of words match the stop words themselves
            # Ex: "annual" was stemmed to "an", and thus "an" appeared in the valid_words dictionary, so when we 
            # see "an" in the word token list, it doesn't get removed by the subsequent line
            new_words = [word.lower() for word in new_words if word in valid_words.keys()]
            # new_words should contain all the words from new_words that are present in valid_words
            tf_scores[num] = {word: 0.0 for word in valid_words}
            frequencies = Counter(new_words)
            for word in new_words:
                tf_scores[num][word] = frequencies[word]/len(new_words)
    return tf_scores


def compute_idf_new(dir_path, valid_words, frequencies, tf_keys):
    '''
    IDF is inverse document frequency, defined as 
    (# of total documents)/(# of occurrences of the word across all documents)
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param frequencies: the return value of the compute_total_word_frequencies
                        function, see above
    :param tf_keys: the numbers of all the documents in the training set
    :return: a dictionary where keys = words and the values = idf scores 
            for that term
    '''
    num_docs = len([name for name in os.listdir(dir_path) if os.path.isfile(dir_path + "\\" + name)])
    idf_scores = {word: 0.0 for word in valid_words.keys()}
    for key in idf_scores.keys():
        idf_scores[key] = np.log(num_docs/(frequencies[key] + 1))
    return idf_scores
            


def compute_tf_idf_by_label(tf_idf, prior_probs, number_labels):
    '''
    This function will compute the total tf_idf score for
    each individual label
    :param tf_idf: a dictionary where keys = number of document and values = 
                    dictionary where keys = words and values = the tf_idf score 
                    of that word in that document
    :param prior_probs: a dictionary where keys = labels and values = the prob
                        of seeing that label (only used so I can grab the unique
                        labels for the document set)
    :param number_labels: a dictionary where the keys = numbers of a document and values
                            = the set of labels associated with it
    :return: a dictionary where keys = labels and values = sum of all tf-idf scores for
            all words that are in that label
    '''
    total_tf_by_label = {label: 0.0 for label in prior_probs.keys()}
    for num, vector in tf_idf.items():
        labels = number_labels[num]
        for l in labels:
            total_tf_by_label[l] += sum(list(vector.values()))
    return total_tf_by_label


def get_valid_words(dir_path, stop_words, st):
    '''
    Utility function that determines the set of valid words 
    to be used for classification and probability calculation
    :param dir_path: a path to the directory containing 
                    all the training samples
    :param stop_words: a set of words like "the", "and", etc
                        that should be stripped out of any computations
    :param st: Lancaster Stemmer object 
    :return: a Python dictionary where the keys = valid words and the 
            values = True, so we can use "key in dict" for future access
            in guaranteed constant time
    '''
    valid_words = defaultdict(bool)
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            words = nltk.word_tokenize(content)
            new_words = [word.lower() for word in words]
            new_words = [word.lower() for word in new_words if word.isalpha()]
            new_words = [word.lower() for word in new_words if word not in stop_words]
            new_words = [st.stem(word) for word in new_words]
            new_words = set(new_words)
            for word in new_words:
                valid_words[word] = True
    return valid_words
            


def add_labels_to_samples(filename):
    '''
    This function iterates over the file containing all 
    labels for each numbered sample, and maps them together with
    a dictionary
    :param filename: path to the file with all the labels in it (assumes
                    the file is located in this directory)
    :return: a dictionary with keys = number of the training sample and
            values = the set of labels associated with it
    '''
    number_labels = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            if line[0:4] == "test":
                continue
            terms = line.split()    # Contains the number of the sample, and the labels
            num = int(terms[0][9:len(terms[0])])  # Test number, so we can map this back to the proper label(s) later on
            for term in terms[1:len(terms)]:
                if num not in number_labels:
                    number_labels[num] = [term]
                else:
                    number_labels[num].append(term)
    return number_labels


def compute_prior_probabilities(number_labels):
    '''
    This function will compute the prior probabilities
    P(y) = probability of seeing a label with a sample. 
    Note: since many samples have multiple labels, these prior
    probabilites will sum to > 1
    :param number_labels: dictionary where keys = number of training sample
                            and value = the list of labels associated with it
    :return: a dictionary where keys = the label and value = probability of seeing
            that label in the document list
    '''
    prior_probs = defaultdict(float)
    i = 0
    for num, labels in number_labels.items():
        for l in labels:
            if not prior_probs[l]:
                prior_probs[l] = 1
            else:
                prior_probs[l] += 1
        i += 1
    for label, freq in prior_probs.items():
        prior_probs[label] /= i
    return prior_probs


def compute_conditional_probabilities(tf_idf, valid_words, prior_probs, number_labels, tf_idf_by_label):
    '''
    This function will take each unique term found in valid_words
    and compute a distribution (Gaussian) for each one by finding the
    mean and standard deviation of the tf-idf scores for each document
    For this to work properly, I need the keys to be indices of the tf_idf
    array. They need to line up properly, otherwise this won't work, so each
    each index of the tf_idf vector maps to the same word
    
    :param tf_idf: a dictionary where keys = document # and values = dictionary
                    where keys = words and values = the tf_idf score of that word
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param prior_probs: a dictionary where keys = labels and values = the prob
                        of seeing that label (only used so I can grab the unique
                        labels for the document set)
    :param number_labels: a dictionary where the keys = number of the sample and 
                        labels = the list of labels associated with it
    :param tf_idf_by_label: a dictionary where keys = labels and values = total of the 
                        tf_idf scores of all words in documents with that label
    :return: a dictionary where the keys = labels and values = dictionary where keys = words
            and values = the tf_idf score for that word given that label
    '''
    # Keys = words, so populate a dictionary first
    conditional_probs = {label: {word: 0.0 for word in valid_words} for label in prior_probs.keys()}
    for num, vector in tf_idf.items():
        labels = number_labels[num]
        for label in labels:
            for word, score in vector.items():
                conditional_probs[label][word] += score
    for label, value in conditional_probs.items():
        for word,score in value.items():
            conditional_probs[label][word] /= (tf_idf_by_label[label] + len(valid_words))
            conditional_probs[label][word] += (1/(tf_idf_by_label[label] + len(valid_words)))
    return conditional_probs


def compute_tf_distributions(dir_path, valid_words, st, stop_words, number_labels, label_list):
    '''
    This function creates one "mega document" for each class and computes
    the tf scores of that document
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param st: Lancaster Stemmer object
    :param stop_words: set of words like "the" and "and" to remove from computations
    :param number_labels: dictionary where keys = document # and values = the set of 
                            labels associated with those labels
    :param label_list: list of all the unique labels
    :return: a dictionary where keys = labels and values = dictionary where keys 
            = words and values = the tf score of that word in the "mega-document"
            of that label 
            AND 
            a dictionary where keys = labels and values = dictionary where keys = words 
            and values =  the number of documents with that label in which that word appears
    '''
    tf = {label: {word: 10**-10 for word in valid_words} for label in label_list}
    label_frequencies = {label: {word: 0 for word in valid_words} for label in label_list}
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4])
            labels = number_labels[num]
            words = nltk.word_tokenize(content)
            words = [st.stem(word) for word in words]
            new_words = [word.lower() for word in words if word not in stop_words]
            new_words = [word.lower() for word in new_words if word in valid_words.keys()]
            frequencies = Counter(new_words)
            unique_words = set(new_words)
            for l in labels:
                for word in unique_words:
                    tf[l][word] += (frequencies[word]/len(new_words))
                    label_frequencies[l][word] += 1
    return [tf, label_frequencies]


def compute_idf_distributions(dir_path, valid_words, number_labels, label_list, frequencies):
    '''
    IDF is inverse document frequency, defined as 
    (# of total documents)/(# of occurrences of the word across all documents)
    This function will compute the idf score of each word across each label
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param number_labels: dictionary where keys = number of document and labels = 
                            the set of labels associated with that document
    :param label_list: list of all unique labels in the dataset
    :param frequencies: dictionary where keys = labels and values = Counter object
                        with frequencies of all terms in the label's "mega document"
    :return: a dictionary where keys = labels and values = dictionary where keys
            = words and values = idf score of that word
    '''
    idf_scores = {label: {word: 0.0 for word in valid_words} for label in label_list}
    label_counts = {label: 0 for label in label_list}
    for file in os.listdir(dir_path):
        num = int(file[0:len(file) - 4])
        labels = number_labels[num]
        for l in labels:
            label_counts[l] += 1
    for label, vector in idf_scores.items():
        for word in vector.keys():
            idf_scores[label][word] = np.log(label_counts[label]/(frequencies[label][word] + 1))
    return idf_scores


def compute_tf_idf_distributions(tf, idf):
    '''
    This function will compute the tf_idf score, grouped by label
    :param tf: Dictionary where keys = labels and values = dictionary
                where keys = words and values = tf score
    :param idf: Dictionary where keys = labels and values = dictionary
                where keys = words and values = idf score
    :return: Dictionary where keys = labels and values = dictionary
                where keys = words and values = tf_idf score
    '''
    tf_idf = {label: {word: 0.0 for word in valid_words} for label in tf.keys()}
    for label, vector in tf.items():
        for word, value in vector.items():
            tf_idf[label][word] = tf[label][word] * idf[label][word]
    return tf_idf


if __name__ == '__main__':
    start = float(datetime.datetime.utcnow().timestamp())
    st = LancasterStemmer() 
    words = ['where', 'whey', 'wheat', 'when', 'wheaty']
    words = nltk.pos_tag(words)
    # print(words)
    stems = [st.stem(word) for word, tag in words]
    # print(stems)
    # lem = WordNetLemmatizer()
    # lemmas = [lem.lemmatize(w, get_wordnet_part_of_speech(t.upper())) for w,t in words]
    # print(lemmas)
    end = float(datetime.datetime.utcnow().timestamp())
    # print(end - start)    

    stop_words = set(stopwords.words('english'))
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\training"
    valid_words = get_valid_words(dir_path, stop_words, st)
    number_labels = add_labels_to_samples("cats.txt")
    prior_probs = compute_prior_probabilities(number_labels)
    tf, frequencies = compute_tf_distributions(dir_path, valid_words, st, 
                                              stop_words, number_labels, 
                                              prior_probs.keys())
    idf = compute_idf_distributions(dir_path, valid_words, number_labels, prior_probs.keys(),
                                        frequencies)
    tf_idf = compute_tf_idf_distributions(tf, idf)

    successes = 0
    i = 0
    for file in os.listdir(dir_path):
        filepath = dir_path + '\\' + file
        num = int(file[0:len(file) - 4])
        text = vectorize_text(stop_words, valid_words, st, filepath)
        computed_labels = naive_bayes(tf_idf, text, prior_probs)
        successes += bayes_accuracy_model(num, number_labels, computed_labels, 10)
        i += 1
    print(successes, i)


