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
import datetime
from operator import itemgetter
import preprocessing as pp


def weight_normalized_cnb(complement_probs_normalized, vectorized_text, prior_probs):
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
            conditional += (freq[word] * complement_probs_normalized[label][word])
        labels.append((label, np.exp(conditional)))
    return sorted(labels, key=itemgetter(1))


def complement_naive_bayes(complement_probs, vectorized_text, prior_probs):
    '''
    :param complement_probs: dictionary where key = label and values = dictionary where
                            keys = words and values = (# of times word w appears in docs
                            NOT labeled l)/(# of words in documents NOT labeled l)
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
        labels.append((label, np.exp(conditional)))
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


def compute_precision_recall(computed_label_set, number_labels, label_list):
    precision = {label: 0.0 for label in label_list}
    recall = {label: 0.0 for label in label_list}
    precision_denom = {label: 0.0 for label in label_list}
    recall_denom = {label: 0.0 for label in label_list}
    for num in computed_label_set.keys():
        computed_labels = computed_label_set[num]
        actual_labels = number_labels[num] 
        computed_labels = computed_labels[:len(actual_labels)]
        for l in computed_labels:  # Positive prediction
            precision_denom[l] += 1
            if l in actual_labels: # True positive
                precision[l] += 1
                recall[l] += 1
                recall_denom[l] += 1                
            diff = set(actual_labels).difference(set(computed_labels))
            for label in diff:
                recall_denom[label] += 1
    total_precision = sum([v for v in precision.values()])
    total_precision_denom = sum([v for v in precision_denom.values()])
    total_precision /= total_precision_denom
    total_recall = sum([v for v in recall.values()])
    total_recall_denom = sum([v for v in recall_denom.values()])
    total_recall /= total_recall_denom
    return [total_precision, total_recall]


def get_parameters(dir_path, valid_words, number_labels, label_list):
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
    valid_words = pp.get_valid_words(dir_path, stop_words)
    number_labels_training, number_labels_test = pp.add_labels_to_samples("cats.txt")
    prior_probs = pp.compute_prior_probabilities(number_labels_training)
    
    parameters = get_parameters(dir_path, valid_words, number_labels_training, prior_probs.keys())
    
    frequencies = parameters[0]
    total_frequencies = parameters[1]
    idf = parameters[2] 
    total_word_count_by_label = parameters[3]
    total_num_words = parameters[4]


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
    conditional_probs_normalized = {label :{word: 0.0 for word in valid_words} for label in prior_probs.keys()}
    normalize_terms = {label: 0.0 for label in prior_probs.items()}
    for label, vector in complement_probs.items():
        normalize_term_1 = np.sqrt(sum([(complement_probs_normalized[label][word]**2) for word in valid_words]))
        normalize_term_2 = np.sqrt(sum([(conditional_probs[label][word]**2) for word in valid_words]))
        normalize_terms[label] = normalize_term_1
        for word in vector.keys():
            complement_probs_normalized[label][word] /= normalize_term_1
            conditional_probs_normalized[label][word] = conditional_probs[label][word] / normalize_term_2

    successes, earned, bottom_5,i = 0, 0, 0, 0
    computed_label_set = defaultdict(list)
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\test"
    for file in os.listdir(dir_path):
        filepath = dir_path + '\\' + file 
        num = int(file[0:len(file) - 4])
        text = pp.vectorize_text(valid_words, filepath)
        computed_labels = complement_naive_bayes(complement_probs, text, prior_probs)
        # computed_labels = multinomial_naive_bayes(conditional_probs_normalized, text, prior_probs)
        # computed_labels = weight_normalized_cnb(complement_probs_normalized, text, prior_probs)
        suc, e, b5 = bayes_accuracy_model(num, number_labels_test, computed_labels)
        computed_label_set[num] = [x for x,y in computed_labels]
        # MNB with doc length normalization, IDF: 86.10% accuracy (2599.288708513709), 1548 "Earn" labels
        # CNB with doc length normalization, IDF: 90.02% accuracy(2717.798340548341), 1130 "Earn" labels
        # However, this approach results in conditional terms that don't make much sense for precision or recall
        # WCNB with doc length normalization, IDF: 87.72% accuracy (2648.141955266955), 1542 "Earn" labels
        
        # Perhaps the reason that TF doesn't lead to improvements with this is because we already stripped out the 
        # stop words, which would be affected the most by this technique

        successes += suc
        earned += e
        bottom_5 += b5
        i += 1
    print(successes, earned, bottom_5, i)

    precision, recall = compute_precision_recall(computed_label_set, number_labels_test, prior_probs.keys())
    print(precision, recall)


