from collections import defaultdict, Counter, OrderedDict
import os.path
import numpy as np
import os
import nltk


def accuracy_model(num, number_labels, labels):
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
    return [successes, earned, bottom_5_times]


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


def get_valid_words(dir_path, stop_words, label_list, number_labels):
    '''
    Utility function that determines the set of valid words
    to be used for classification and probability calculation
    :param dir_path: a path to the directory containing
                    all the training samples
    :param stop_words: a set of words like "the", "and", etc
                        that should be stripped out of any computations
    :param label_list: a list of all the valid labels in the corpus
    :param number_labels: a dictionary where keys = number of document
                        and values = the list of labels that this document
                        has
    :return: a dictionary where keys = labels and values = a dictionary
            where keys = all the unique words that exist in documents
            with this label
    '''
    total_valid_words = defaultdict(bool)
    valid_words = {label: defaultdict(bool) for label in label_list}
    for file in os.listdir(dir_path):
        with open(dir_path + '\\' + file, "r") as f:
            content = f.read()
            num = int(file[0:len(file) - 4])
            words = nltk.word_tokenize(content)
            new_words = [word.lower() for word in words]
            new_words = [word.lower() for word in new_words if word.isalpha()]
            new_words = [word.lower() for word in new_words if word not in stop_words]
            new_words = set(new_words)
            labels = number_labels[num]
            for l in labels:
                for word in new_words:
                    total_valid_words[word] = True
                    valid_words[l][word] = True
    return valid_words, OrderedDict(sorted(total_valid_words.items(), key=lambda t: t[0]))


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


def compute_precision_recall(computed_label_set, number_labels, label_list):
    '''
    '''
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


def get_df(dir_path, number_labels, label_list, valid_words):
    '''
    :param dir_path: path to the directory containing all
                    the training files
    :param number_labels: dictionary where keys = number of document
                            and values = the set of labels associated
                            with that document
    :param label_list: list of all the labels in the training set
    :return: a dictionary where keys = labels and values = a
            dictionary where keys = words and values = the
            number of documents with label l that word appears in
    '''
    df = {label: {word: 0.0 for word in valid_words.keys()} for label in label_list}
    for file in os.listdir(dir_path):
        filepath = dir_path + '\\' + file
        with open(filepath, "r") as f:
            num = int(file[0:len(file) - 4])
            vector = vectorize_text(valid_words, filepath)
            freq = Counter(vector)
            labels = number_labels[num]
            for l in labels:
                for word in freq.keys():
                    if not df[l][word]:
                        df[l][word] = 1
                    else:
                        df[l][word] += 1
    return df


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


