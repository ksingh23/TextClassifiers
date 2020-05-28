from collections import defaultdict
import os.path
import os
import nltk


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