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
import preprocessing as pp
from nltk.corpus import stopwords
import featureselection as fs
from operator import itemgetter


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
    doc_denom = 0
    freq = Counter(vectorized_text)
    '''
    for word in freq.keys():
        for label in prior_probs.keys():
            doc_denom += (np.log(prior_probs[label]) + (freq[word]/len(vectorized_text) * complement_probs[label][word]))
    print(doc_denom)
    '''
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


if __name__ == '__main__':
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\TextClassifiers\\training"
    stop_words = set(stopwords.words('english'))
    number_labels_training, number_labels_test = pp.add_labels_to_samples("cats2.txt")
    # valid_words = fs.most_useful_features("cdmfeatures.txt")
    prior_probs = compute_prior_probabilities(number_labels_training)
    valid_words_by_label, valid_words = pp.get_valid_words(dir_path, stop_words, prior_probs.keys(), number_labels_training)
    parameters = pp.get_parameters(dir_path, valid_words, number_labels_training, prior_probs.keys())
    
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

    # Removing the stemmer actually improves accuracy on test set, who knew
    successes, earned, bottom_5,i = 0, 0, 0, 0
    computed_label_set = defaultdict(list)
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\TextClassifiers\\test"
    for file in os.listdir(dir_path):
        filepath = dir_path + '\\' + file 
        num = int(file[0:len(file) - 4])
        text = pp.vectorize_text(valid_words, filepath)
        computed_labels = complement_naive_bayes(complement_probs, text, prior_probs)
        # computed_labels = multinomial_naive_bayes(conditional_probs_normalized, text, prior_probs)
        # computed_labels = weight_normalized_cnb(complement_probs_normalized, text, prior_probs)
        suc, e, b5 = pp.accuracy_model(num, number_labels_test, computed_labels)
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

    precision, recall = pp.compute_precision_recall(computed_label_set, number_labels_test, prior_probs.keys())
    print(precision, recall)


