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


def weight_normalized_cnb(complement_probs, idf, vectorized_text, 
                          prior_probs):
    '''
    :param complement_probs: dictionary where key = label and values = dictionary where
                            keys = words and values = (# of times word w appears in docs
                            NOT labeled l)/(# of words in documents NOT labeled l)
    :param idf: dictionary where keys = words and values = (total # docs)/(# of docs in 
                which we see that word)
    :param vectorized_text: words from text that are in valid_words
    :param prior_probs: dictionary where keys = labels and values = the probability
                        of seeing that label in the dataset
    '''
    labels = []
    freq = Counter(vectorized_text)
    for label in prior_probs.keys():
        conditional = 0.0
        for word in freq.keys():
            conditional += (freq[word] * complement_probs[label][word])
        labels.append((label, conditional))
    return sorted(labels, key=itemgetter(1))


def complement_naive_bayes(complement_probs, idf, vectorized_text, prior_probs_normalized, prior_probs):
    '''
    :param complement_probs: dictionary where key = label and values = dictionary where
                            keys = words and values = (# of times word w appears in docs
                            NOT labeled l)/(# of words in documents NOT labeled l)
    :param vectorized_text: words from text that are in valid_words
    :param prior_probs: dictionary where keys = labels and values = the probability
                        of seeing that label in the dataset
    '''
    labels = []
    doc_denom = 0
    freq = Counter(vectorized_text)
    '''
    for word in freq.keys():
        for label in prior_probs.keys():
            doc_denom += (np.log(prior_probs[label]) + (freq[word]/len(vectorized_text) * complement_probs[label][word]))
    print(doc_denom)
    '''
    normalization_term = np.sqrt(sum([freq[word]**2 for word in freq.keys()]))
    for label in prior_probs.keys():
        prob = prior_probs_normalized[label]
        conditional = 0.0
        for word in freq.keys():
            conditional += (freq[word] * complement_probs[label][word])
        prob -= conditional
        labels.append((label, conditional))
    return sorted(labels, key=itemgetter(1))


def multinomial_naive_bayes(conditional_probs, vectorized_text, prior_probs):
    '''
    :param conditional_probs: dictionary where keys = labels and values = dictionary where
                    keys = words and values = P(x|Y)
    :param vectorized_text: words from text that are in valid_words
    :param prior_probs: dictionary where keys = labels and values = the probability
                        of seeing that label in the dataset
    '''
    labels = []
    freq = Counter(vectorized_text)
    for label in prior_probs.keys():
        conditional = 0.0
        for word in vectorized_text:
            if conditional_probs[label][word] != 0.0:
                conditional += (freq[word] * conditional_probs[label][word])
        labels.append((label, conditional))
    return sorted(labels, key=itemgetter(1), reverse=True)


def bayes_accuracy_model(num, number_labels, labels):
    '''
    :param num: the number of the document being checked, so we can check
                the correct labels for it
    :param number_labels: dictionary where keys = number of sample and
                            values = the set of labels associated with
                            that sample
    :param labels: the set of labels computed by Naive Bayes
    rye 0.00012871669455528383
    groundnut-oil 0.00012871669455528383
    cotton-oil 0.00012871669455528383
    castor-oil 0.00012871669455528383
    nkr 0.00012871669455528383
    sun-meal 0.00012871669455528383
    '''
    sample_labels = number_labels[num]
    successes = 0
    earned = 0
    bottom_5_times = 0
    bottom_5 = ['rye', 'groundnut-oil', 'cotton-oil', 'castor-oil', 'nkr', 'sun-meal']
    computed_labels = [x for x,y in labels]
    if "earn" in computed_labels[:3]:
        earned += 1
    computed_labels_trim = computed_labels[:len(sample_labels)]
    for label in bottom_5:
        if label in computed_labels[:5]:
            bottom_5_times += 1
            break
    if all(x in computed_labels_trim for x in sample_labels):
        successes += 1
    else:
        print(num)
        print(sample_labels, labels[:10])
        diff = set(sample_labels).difference(set(computed_labels_trim))
        score = len(computed_labels_trim) - len(diff)
        score /= len(computed_labels_trim)
        if len(diff) < len(computed_labels_trim):
            successes += score
    return [successes,earned, bottom_5_times]


def vectorize_text(valid_words, filepath):
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
    frequencies = {label: {word: 0 for word in valid_words} for label in label_list}
    total_frequencies = {word:0 for word in valid_words}
    for label, vector in mega_docs.items():
        freq = Counter(vector)
        for word in freq.keys():
            frequencies[label][word] += freq[word]
            total_frequencies[word] += freq[word]
    return [frequencies, total_frequencies]


def compute_word_frequencies(dir_path, valid_words, number_labels, label_list):
    '''
    This function will iterate over the documents and compute the frequencies of 
    the words by label and in total
    :param dir_path: a path to the directory containing all the training samples
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param number_labels: dictionary where keys = document # and values = the set of 
                            labels associated with those labels
    :param label_list: list of all the unique labels
    :return: a dictionary where keys = labels and values = dictionary where keys 
            = words and values = the frequencies of that word in documents with that label
            AND 
            a dictionary where keys = words and values = the total # of occurrences of
            that word
            AND
            a dictionary where keys = words and values = the idf score for that word
            AND 
            a dictionary where keys = labels and values = the total # of words associated with that label
            AND
            the total # of valid words in the entire corpus
    '''
    words_by_doc_num = defaultdict()
    idf = {word: 0.0 for word in valid_words}
    total_num_words = 0
    total_word_count_by_label = {label: 0 for label in label_list}    
    i = 0
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4])
            labels = number_labels[num]
            words = nltk.word_tokenize(content)
            new_words = [word.lower() for word in words]
            new_words = [word.lower() for word in new_words if word in valid_words]
            total_num_words += len(new_words)
            freq = Counter(new_words)
            words_by_doc_num[num] = freq
            for word in freq.keys():
                idf[word] += 1
            for l in labels:
                total_word_count_by_label[l] += len(new_words)
            i += 1
    for word in idf.keys():
        idf[word] = 1 + np.log(i/(idf[word]+1))      
    frequencies = {label: {word: 0.0 for word in valid_words} for label in label_list} 
    total_frequencies = {word: 0 for word in valid_words}
    for num in words_by_doc_num.keys():
        freq = words_by_doc_num[num]
        labels = number_labels[num]
        normalization_term = np.sqrt(sum([score**2 for word, score in freq.items()]))
        for l in labels:
            total_word_count_by_label[l] += len(new_words)
            for word in freq.keys():
                term_to_add = freq[word] * idf[word]
                frequencies[l][word] += (term_to_add/normalization_term)
                total_frequencies[word] += (term_to_add/normalization_term)
    return [frequencies, total_frequencies, idf, total_word_count_by_label, total_num_words]


if __name__ == '__main__':
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\training"
    stop_words = set(stopwords.words('english'))
    valid_words = get_valid_words(dir_path, stop_words)
    number_labels_training, number_labels_test = add_labels_to_samples("cats.txt")
    prior_probs = compute_prior_probabilities(number_labels_training)
    
    parameters = compute_word_frequencies(dir_path, valid_words, number_labels_training, prior_probs.keys())
    
    frequencies = parameters[0]
    total_frequencies = parameters[1]
    idf = parameters[2] 
    total_word_count_by_label = parameters[3]
    total_num_words = parameters[4]

    for label, vector in frequencies.items():
        if label != "earn":
            continue
        print("Label:", label)
        for word, score in sorted(vector.items(), key=itemgetter(1), reverse=True):
            if score == 0.0:
                continue
            print(word, score)

    conditional_probs = {label: {word: 0.0 for word in valid_words} for label in prior_probs.keys()}
    complement_probs = {label: {word: 0.0 for word in valid_words} for label in prior_probs.keys()}
    for label, vector in conditional_probs.items():
        denom = total_num_words - total_word_count_by_label[label] + len(valid_words.keys())
        for word in vector.keys():
            mod_cond_freq = frequencies[label][word] + 1
            mod_comp_freq = (total_frequencies[word] - frequencies[label][word]) + 1
            conditional_probs[label][word] = np.log(mod_cond_freq/(total_word_count_by_label[label] + len(valid_words.keys())))
            complement_probs[label][word] = np.log(mod_comp_freq/denom)

    complement_probs_normalized = {label: {word: complement_probs[label][word] for word in valid_words} 
                                   for label in prior_probs.keys()}
    for label, vector in complement_probs.items():
        normalize_term = np.sqrt(sum([(complement_probs_normalized[label][word]**2) for word in valid_words]))
        for word in vector.keys():
            complement_probs_normalized[label][word] /= normalize_term

    # Removing the stemmer actually improves accuracy on test set, who knew
    successes, earned, bottom_5,i = 0, 0, 0, 0
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\test"
    for file in os.listdir(dir_path):
        filepath = dir_path + '\\' + file 
        num = int(file[0:len(file) - 4])
        text = vectorize_text(valid_words, filepath)
        # computed_labels = complement_naive_bayes(complement_probs, idf, text, prior_probs_normalized, prior_probs)
        # computed_labels = multinomial_naive_bayes(conditional_probs, text, prior_probs)
        computed_labels = weight_normalized_cnb(complement_probs_normalized, idf, text, 
                                               prior_probs)
        suc, e, b5 = bayes_accuracy_model(num, number_labels_test, computed_labels)
        # Even with using conditional_probs, earn appears in 1773/3019 samples
        
        # CNB brought earn labels down to 1170/3019, which is the best improvement so far
             
        # Multinomial Naive Bayes: 84.09% (2538.627561327562) accuracy on test set (????), 1773 "Earn" labels
        # Complement Naive Bayes: 85.09% (2568.913203463204) accuracy on test set, 1687 "Earn" labels
        # Weight Normalized CNB w/ TF-IDF transformation: 76.35% (2304.986291486292), 2131 "Earn" labels
        # (I FORGOT HOW I GOT THIS AHHHHH)
        # CNB with IDF transformation: 86.76% (2619.324711399711), 1321 "Earn" labels
        # MNB with IDF transformation: 84.87% (2562.234704184704), 1521 "Earn" labels
        
        # MNB with doc length normalization, IDF: 86.10% accuracy (2599.288708513709), 1548 "Earn" labels
        # CNB with doc length normalization, IDF: 86.93% accuracy(2624.394769119769), 1312 "Earn" labels
        # WCNB with doc length normalization and IDF^2: 87.72% accuracy (2648.141955266955), 1542 "Earn" labels
        
        # Perhaps the reason that TF doesn't lead to improvements with this is because we already stripped out the 
        # stop words, which would be affected the most by this technique

        successes += suc
        earned += e
        bottom_5 += b5
        i += 1
    print(successes, earned, bottom_5, i)


