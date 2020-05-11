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
    


def compute_avg_tf_idf_by_label(tf_idf, label_list, number_labels):
    '''
    The goal of this function is to find a tf_idf vector
    representation of each unique label. This is done by summing the
    vectors for each document with a specific label, and dividing by the #
    of these documents
    :param tf_idf: a dictionary where keys = document # and values = 
                    a dictionary where keys = word and value = the tf_idf
                    score for that term
    :param label_list: list of the unique labels on the documents
    :param number_labels: dictionary where the key = document # and the value
                            = the set of labels associated with those samples
    :return: a dictionary where the keys = labels and values = dictionary where the 
                keys = words and values = the average tf_idf score for all documents 
                with that label
    '''
    avg_tf_idf = {label: defaultdict(float) for label in label_list}
    label_count = {label: 0  for label in label_list}
    for num, scores in tf_idf.items():
        labels = number_labels[num]
        for l in labels:
            label_count[l] += 1
        for word, value in scores.items():
            for l in labels:
                if word not in avg_tf_idf.keys():
                    avg_tf_idf[l][word] = tf_idf[num][word]
                else:
                    avg_tf_idf[l][word] += tf_idf[num][word]
    for label, scores in avg_tf_idf.items():
        for word, value in scores.items():
            avg_tf_idf[label][word] /= label_count[label]
    return avg_tf_idf


def tf_idf_for_one_doc(filepath):
    '''
    This function takes a single document and computes the tf_idf score for it,
    so we don't need to enter a directory like we do to compute prior and conditional
    probabilities for training
    :param filepath: the path to the file we are computing tf_idf for
    '''
    with open(filepath, "r") as f:
        content = f.read()
        words = nltk.word_tokenize(content)
        new_words = [word.lower() for word in words if word.isalpha()]
        new_words = [word.lower() for word in new_words if word not in stop_words]
        frequencies = Counter(new_words)


def naive_bayes(prior_probs, conditional_probs, vectorized_text, label_list):
    lab = []
    # Instead of working with products of what could be very small floats
    # we work with the log of all the probabilities, to deal with sums that
    # end up being less negligible than before
    # P(Y | x) = P(Y) * P(x | Y)/P(x)
    # P(x) = sum of P(y_i)*(P(x | y_i)) for all i in the list of labels
    for label in label_list:
        bayes_prob = 0
        for word in vectorized_text:
            # term_total_prob = sum([prior_probs[label] * conditional_probs[label][word] for label in label_list])
            prior_prob = prior_probs[label]
            conditional_prob = conditional_probs[label][word]
            bayes_prob += (math.log(conditional_prob) + math.log(prior_prob))
        # bayes_prob = math.exp(bayes_prob)
        lab.append((label, bayes_prob))
    return sorted(lab, key=itemgetter(1), reverse=True)


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
            


def compute_tf_by_label(tf, prior_probs, number_labels):
    '''
    This function will compute the total tf score for
    each individual label
    :param tf: a dictionary where keys = words and values = a vector of the tf score 
            for each document (so the length of each of these vectors is the length
            of the training set)
    :param prior_probs: a dictionary where keys = labels and values = the prob
                        of seeing that label (only used so I can grab the unique
                        labels for the document set)
    :param number_labels: a dictionary where the keys = numbers of a document and values
                            = the set of labels associated with it
    :return: a dictionary where keys = labels and values = sum of all tf-idf scores for
            all words that are in that label
    '''
    total_tf_by_label = {label: 0.0 for label in prior_probs.keys()}
    for word, vector in tf.items():
        for i in range(len(vector)):
            if vector[i] == 0.0:
                continue
            labels = number_labels[i+1]
            for l in labels:
                total_tf_by_label[l] += vector[i]
    return total_tf_by_label


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


def compute_conditional_probabilities(tf_idf, valid_words, prior_probs, number_labels, tf_by_label):
    '''
    This function will take each unique term found in valid_words
    and compute a distribution (Gaussian) for each one by finding the
    mean and standard deviation of the tf-idf scores for each document
    :param tf_idf: a dictionary where keys = terms and values = vectors
                    of len(# of training samples) where the ith entry 
                    corresponds to the tf_idf score of that word in
                    training sample i + 1 (so tf_idf['word'][10] would 
                    represent the tf_idf score of the term 'word' in
                    document # 11 in the training set)
    :param valid_words: a dictionary where the keys are all the unique, valid
                        terms are present in the text file
    :param prior_probs: a dictionary where keys = labels and values = the prob
                        of seeing that label (only used so I can grab the unique
                        labels for the document set)
    :param number_labels: a dictionary where the keys = number of the sample and 
                        labels = the list of labels associated with it
    :param tf_by_label: a dictionary where keys = labels and values = total of the 
                        tf_idf scores of all words in documents with that label
    :return: a dictionary where the keys = terms from valid_words and 
                values = another dictionary where keys = labels and values
                = the mean, std. dev of the occurrences of that word in documents
                with the key label
    '''
    # Keys = words, so populate a dictionary first
    conditional_probs = {word: {l: 0.0 for l in prior_probs.keys()} for word in valid_words}
    for term, vector in tf_idf.items():
        for i in range(len(vector)):
            if vector[i] == 0:   # term not present in this document
                continue
            label_list = number_labels[i+1]
            for l in label_list:
                conditional_probs[term][l] += vector[i]
    for word, value in conditional_probs.items():
        for label,score in value.items():
            conditional_probs[word][label] /= (tf_by_label[label] + len(valid_words))
            conditional_probs[word][label] += (1/(tf_by_label[label] + len(valid_words)))
    return conditional_probs


if __name__ == '__main__':
    start = float(datetime.datetime.utcnow().timestamp())
    st = LancasterStemmer() 
    words = ['where', 'whey', 'wheat', 'when', 'wheaty']
    words = nltk.pos_tag(words)
    print(words)
    stems = [st.stem(word) for word, tag in words]
    print(stems)
    # lem = WordNetLemmatizer()
    # lemmas = [lem.lemmatize(w, get_wordnet_part_of_speech(t.upper())) for w,t in words]
    # print(lemmas)
    end = float(datetime.datetime.utcnow().timestamp())
    print(end - start)    

    stop_words = set(stopwords.words('english'))
    # print(stop_words, "\n")
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\Text Classifiers\\MiniTrainingSet"
    valid_words = get_valid_words(dir_path, stop_words, st)
    frequencies = compute_total_word_frequencies(dir_path, valid_words, st)
    # 150 unique words without the stemmer
    # 142 with it
    # print(valid_words.keys(), "\n")
    # Valid words does not strip out stop words. Why is that?
    tf = compute_tf_new(dir_path, valid_words, st, stop_words)
    idf = compute_idf_new(dir_path, valid_words, frequencies, tf.keys())
    tf_idf = compute_tf_idf(tf, idf)
    '''
    for num,value in tf_idf.items():
        print("Document #", num)
        for key, v in sorted(value.items(), key=itemgetter(1), reverse=True):
            if v == 0.0:
                continue
            print(key, v)
        print("\n")
    '''

    number_labels = add_labels_to_samples("cats1.txt")
    prior_probs = compute_prior_probabilities(number_labels)
    avg_tf_idf = compute_avg_tf_idf_by_label(tf_idf, prior_probs.keys(), number_labels)
    for label, scores in avg_tf_idf.items():
        print("Label: ", label)
        for key, value in sorted(scores.items(), key=itemgetter(1), reverse=True):
            if value == 0.0:
                continue
            print(key, value)
        print("\n")

    prior_probs = compute_prior_probabilities(number_labels)
    tf_by_label = compute_tf_by_label(tf, prior_probs, number_labels)
    conditional_probs = compute_conditional_probabilities(tf_idf, valid_words, prior_probs, number_labels, tf_by_label)
    #for key, value in tf_by_label.items():
    #    print(key, value)
    for key, vector in conditional_probs.items():
        print("Key: " + key)
        for k,v in vector.items():
            print(k, v)
        print("\n")


