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
from nltk.tokenize import word_tokenize
import datetime
from operator import itemgetter


def naive_bayes(conditional_probs, complement_probs, frequencies, vectorized_text, prior_probs):
    '''
    :param conditional_probs: dictionary where keys = labels and values = dictionary where
                    keys = words and values = P(x|Y)
    :param complement_probs: dictionary where key = label and values = dictionary where
                            keys = words and values = (# of times word w appears in docs
                            NOT labeled l)/(# of words in documents NOT labeled l)
    :param frequencies: dictionary where keys = labels and values = dictionary where
                    keys = words and values = frequencies of that word given that label
    :param vectorized_text: words from text that are in valid_words
    :param prior_probs: dictionary where keys = labels and values = the probability
                        of seeing that label in the dataset
    '''
    labels = []
    for label in prior_probs.keys():
        prob = np.log(prior_probs[label])
        conditional = 0.0
        for word in vectorized_text:
            # This is currently outputtng NaN, why is that?
            if conditional_probs[label][word] != 0.0:
                conditional += (frequencies[label][word] * np.log(conditional_probs[label][word]))
            if complement_probs[label][word] != 0.0:
                conditional -= (frequencies[label][word] * np.log(complement_probs[label][word]))
        prob += conditional
        labels.append((label, prob))
    return sorted(labels, key=itemgetter(1), reverse=True)


def bayes_accuracy_model(num, number_labels, labels):
    '''
    :param num: the number of the document being checked, so we can check
                the correct labels for it
    :param number_labels: dictionary where keys = number of sample and
                            values = the set of labels associated with
                            that sample
    :param labels: the set of labels computed by Naive Bayes
    '''
    sample_labels = number_labels[num]
    successes = 0
    earned = 0
    computed_labels = [x for x,y in labels]
    if "earn" in computed_labels[:3]:
        earned += 1
    computed_labels_trim = computed_labels[:len(sample_labels)]
    if all(x in computed_labels_trim for x in sample_labels):
        successes += 1
    else:
        print(sample_labels, computed_labels[:10])
        diff = set(sample_labels).difference(set(computed_labels_trim))
        if len(diff) < len(computed_labels_trim):
            successes += (len(diff)/len(computed_labels_trim))
    return [successes,earned]


def vectorize_text(stop_words, valid_words, filepath):
    '''
    This function removes non valid words from the text to put it into
    the Naive Bayes classifier
    :param stop_words: a set of words like "the", "and", etc
                        that should be stripped out of any computations
    :param valid_words: dictionary where keys = valid words in the corpus
    :param filepath: path to the text file
    :return: a vector of text stripped of stop words and non-valid words
    '''
    with open(filepath, "r") as f:
        content = f.read()
        words = nltk.word_tokenize(content)
        words = [word.lower() for word in words]
        new_words = [word.lower() for word in words if word in valid_words]
    return new_words


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


def compute_total_word_frequencies(dir_path, valid_words):
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
            new_words = [word.lower() for word in words if word not in stop_words]
            new_words = [word.lower() for word in new_words if word in valid_words.keys()]
            new_words = set(new_words)
            for word in new_words:
                frequencies[word] += 1
    return frequencies


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
            new_words = [word.lower() for word in words]
            new_words = [word.lower() for word in new_words if word.isalpha()]
            new_words = [word.lower() for word in new_words if word not in stop_words]
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
            AND
            the same, but with the test samples. Keep them separate for easy
            access later
    '''
    number_labels_training = defaultdict(list)
    number_labels_test = defaultdict(list)
    with open(filename, "r") as f:
        for line in f:
            terms = line.split()
            if line[0:4] == "test":
                num = int(terms[0][5:len(terms[0])])  # Test number, so we can map this back to the proper label(s) later on
                number_labels_test[num] = terms[1:]
            else:
                num = int(terms[0][9:len(terms[0])])  
                number_labels_training[num] = terms[1:]
    return [number_labels_training, number_labels_test]


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


def word_vectors_for_mega_docs(dir_path, valid_words, number_labels, label_list):
    '''
    This function returns a list of all the words from documents
    of each label
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param number_labels: dictionary where keys = document # and values = the set of 
                            labels associated with those labels
    :param label_list: list of all the unique labels
    :return: a dictionary where keys = labels and values = a dictionary where 
            keys = words and values = a vector with all the valid words in 
            documents with that label
    '''
    mega_docs = {label: [] for label in label_list}
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4])
            labels = number_labels[num]
            words = nltk.word_tokenize(content)
            words = [word.lower() for word in words]
            new_words = [word.lower() for word in words if word in valid_words]
            for l in labels:
                mega_docs[l] += new_words
    return mega_docs


def compute_tf_distributions(dir_path, valid_words, number_labels, label_list):
    '''
    This function creates one "mega document" for each class and computes
    the tf scores of that document
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
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
    tf = {label: {word: 0.0 for word in valid_words} for label in label_list}
    label_frequencies = {label: {word: 0 for word in valid_words} for label in label_list}
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4])
            labels = number_labels[num]
            words = nltk.word_tokenize(content)
            new_words = [word.lower() for word in words]
            new_words = [word.lower() for word in new_words if word in valid_words]
            frequencies = Counter(new_words)
            unique_words = set(new_words)
            other_labels = set(label_list).difference(labels)
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
            # Only one occurrence of lin-oil, and thus, the "mega document" is just
            # the single document itself. 
            idf_scores[label][word] = 1 + np.log(label_counts[label]/(frequencies[label][word] + 1))
    return idf_scores, label_counts


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
            # if label == "earn":
            #    print(word, tf[label][word], idf[label][word])
            tf_idf[label][word] = tf[label][word] * idf[label][word]
    return tf_idf


def rename_files(dir_path):
    '''
    Utility function designed to rename all files in any directory
    to a .txt file so they can be read from
    :param dir_path: directory of the files to be renamed
    '''
    for file in os.listdir(dir_path):
        filepath = dir_path + '\\' + file 
        os.rename(filepath, filepath+".txt")


def compute_frequencies_by_class(mega_docs, valid_words, label_list):
    '''
    This function computes the frequencies of all words by class. This is done because
    the outright frequencies are needed for Naive Bayes and conditional_probs can easily
    be obtained from this by dividing each entry by the number of elements in each "mega doc"
    :param mega_docs: a dictionary where keys = labels and values = vectors of all the
                        valid words present in documents with that label
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param label_list: list of all unique labels in the dataset
    :return: a dictionary where keys = labels and values = dictionary where keys = words
                and values = frequencies of that word in docs with that label
            AND
            a dictionary where keys = words and values = the total frequency of those words
            all documents throughout the corpus
    '''
    frequencies = {label: {word: 0.0 for word in valid_words} for label in label_list}
    total_frequencies = {word:0 for word in valid_words}
    for label, vector in mega_docs.items():
        freq = Counter(vector)
        for word in frequencies[label].keys():
            if freq[word]:
                frequencies[label][word] += freq[word]
                total_frequencies[word] += freq[word]
    return [frequencies, total_frequencies]


if __name__ == '__main__':
    '''
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\training"
    stop_words = set(stopwords.words('english'))
    valid_words = get_valid_words(dir_path, stop_words)
    number_labels_training, number_labels_test = add_labels_to_samples("cats.txt")
    prior_probs = compute_prior_probabilities(number_labels_training)
    tf, frequencies = compute_tf_distributions(dir_path, valid_words, number_labels_training, 
                                              prior_probs.keys())
    mega_docs = word_vectors_for_mega_docs(dir_path, valid_words, number_labels_training, prior_probs.keys())
    idf, label_counts = compute_idf_distributions(dir_path, valid_words, number_labels_training, prior_probs.keys(),
                                        frequencies)
    tf_idf = compute_tf_idf_distributions(tf, idf)
    '''
    

    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\training"
    stop_words = set(stopwords.words('english'))
    valid_words = get_valid_words(dir_path, stop_words)
    number_labels_training, number_labels_test = add_labels_to_samples("cats.txt")
    prior_probs = compute_prior_probabilities(number_labels_training)
    
    mega_docs = word_vectors_for_mega_docs(dir_path, valid_words, number_labels_training, prior_probs.keys())
    frequencies, total_frequencies = compute_frequencies_by_class(mega_docs, valid_words, prior_probs.keys())
    conditional_probs = {label: {word: 0.0 for word in valid_words} for label in prior_probs.keys()}
    complement_probs = {label: {word: 0.0 for word in valid_words} for label in prior_probs.keys()}
    for label, vector in conditional_probs.items():
        denom = sum([len(v) for v in mega_docs.values()]) - len(mega_docs[label])
        # print(label, len(mega_docs[label]), denom)
        for word in vector.keys():
            conditional_probs[label][word] = frequencies[label][word]/len(mega_docs[label])
            complement_probs[label][word] = (total_frequencies[word] - frequencies[label][word])/denom
    

    '''
    for label, vector in conditional_probs.items():
        print("Label:", label, len(mega_docs[label]))
        for word, score in sorted(vector.items(), key=itemgetter(1), reverse=True):
            if score == 0.0:
                continue
            print(word, score, complement_probs[label][word])
        print("\n")
    '''

    '''
    for key, value in sorted(prior_probs.items(), key=itemgetter(1), reverse=True):
        print(key, prior_probs[key])
    '''

    # Removing the stemmer actually improves accuracy on test set, who knew
    successes, earned = 0, 0
    i = 0
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\test"
    for file in os.listdir(dir_path):
        filepath = dir_path + '\\' + file 
        num = int(file[0:len(file) - 4])
        text = vectorize_text(stop_words, valid_words, filepath)
        computed_labels = naive_bayes(conditional_probs, complement_probs, frequencies, text, prior_probs)
        suc, e = bayes_accuracy_model(num, number_labels_test, computed_labels)
        # Even with using conditional_probs, earn appears in 2936/3019 samples, so we can
        # try CNB again to see if that remedies it.
        successes += suc
        earned += e
        i += 1
    print(successes, earned, i)


