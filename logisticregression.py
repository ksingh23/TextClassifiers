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
from collections import defaultdict, Counter, OrderedDict
import featureselection as fs
import os.path
import os
import nltk
from nltk.corpus import stopwords
import datetime
from operator import itemgetter
import preprocessing as pp
import time


def softmax(x):
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()


def sigmoid(x):
    return .5 * (1 + np.tanh(.5 * x))


def logistic_regression(weights, data, binary_label_vectors, threshold, learning_rate):
    '''
    :param weights: a matrix of len(valid_words) rows and len(label_list)
                    columns, where entry i,j is the weight for word i
                    in documents with label j
    :param data: a matrix of len(valid_words) rows and num_docs columns,
                where entry i,j is the number of occurrences of word i
                in document j
    :param binary_label_vectors: a matrix of len(label_list) rows and
                                num_docs cols, where each entry i,j is
                                1 if document j has label i, and 0 otherwise
    :param threshold: the value at which if all weights only change by less than
                        this value, we claim the model has converged
    :param learning rate: the scaling factor of the gradient to ensure we do not
                            descend too quickly
    :return: the weights matrix, updated after running gradient descent on it 
    '''
    for label in binary_label_vectors.keys():
        print("Training label", label)
        labels = np.asarray(binary_label_vectors[label]).reshape(1, data.shape[1])
        # Labels.shape = 1x5
        num_docs = labels.shape[0]
        weight_vector = np.transpose(weights[label]).reshape(1, data.shape[0])
        converged = False
        k = 1
        num_docs = labels.shape[0]
        while not converged and k < 50:
            converged = True
            print(label, "Iteration:", k)
            start = time.time()
            weight_eval = sigmoid(np.dot(weight_vector, data))
            # weight_eval.shape = 1x5
            print("W^T * x", time.time() - start)
            start = time.time()
            loss = np.subtract(labels, weight_eval)
            # loss.shape = 1x5
            gradient = np.dot(loss, np.transpose(data))
            gradient *= learning_rate
            print("Gradient: ", time.time()-start)
            start = time.time()
            updated_weights = weight_vector + gradient
            delta = weight_vector - updated_weights
            weight_vector = updated_weights
            for i in range(len(delta)):
                for j in range(len(delta[0])):
                    if np.absolute(delta[i][j]) >= threshold:
                        converged = False
            k += 1
            print("Check convergence: ", time.time()-start)
            # gradient.shape = 1x308
        # print("Iterations: ", label, k)
        weights[label] = weight_vector
    return weights


def build_new_cats_file(filename):
    '''
    This is a utility function that will rearrange the elements
    in the cats.txt file so that they are sorted by numbers instead
    '''
    with open(filename, "r") as f:
        training_lines = []
        test_lines = []
        for line in f:
            elems = line.split()
            if line[0:8] == "training":
                num = int(elems[0][9:len(elems[0])])
                training_lines.append((num, elems[1:]))
            else:
                num = int(elems[0][5:len(elems[0])])
                test_lines.append((num, elems[1:]))
    training_lines = sorted(training_lines, key=lambda x: x[0])
    test_lines = sorted(test_lines, key=lambda x: x[0])
    with open("cats2.txt", "w") as f:
        for num, labels in training_lines:
            f.write("training/" + str(num) + " " + " ".join(labels) + "\n")
        for num, labels in test_lines:
            f.write("test/" + str(num) + " " + " ".join(labels) + "\n")


def construct_feature_matrix(dir_path, valid_words):
    '''
    This function will take text vectors of each document
    and put them into a matrix.
    :param dir_path: path to the directory containing all the
                    training files
    :param valid_words: a dictionary where keys are all the 
                        valid words in the corpus
    '''
    feature_matrix = []
    sorted_files = sorted([int(file[0:len(file) - 4])  for file in os.listdir(dir_path)])
    for num in sorted_files:
        filepath = dir_path + "\\" + str(num) + ".txt"
        text_vector = pp.vectorize_text(valid_words, filepath)
        word_freq = {word: 0 for word in valid_words}
        freq = Counter(text_vector)
        for word in freq.keys():
            word_freq[word] = freq[word]
        frequencies = [y for x,y in word_freq.items()]
        feature_matrix.append(frequencies)
    return np.transpose(np.asarray(feature_matrix))


def build_text_vectors(dir_path, valid_words):
    text_vectors = defaultdict()
    sorted_files = sorted([int(file[0:len(file) - 4])  for file in os.listdir(dir_path)])
    for num in sorted_files:
        frequencies = {word: 0 for word in valid_words}
        filepath = dir_path + "\\" + str(num) + ".txt"
        text_vector = pp.vectorize_text(valid_words, filepath)
        freq = Counter(text_vector)
        for word in freq.keys():
            frequencies[word] = freq[word]
        text_vectors[num] = np.asarray([y for x,y in frequencies.items()])
    return text_vectors


def construct_binary_label_vectors(indexed_labels, filename, num_docs):
    '''
    This function will return a dictionary where keys = labels 
    and values = a 2D vector of len(label_list) rows and len(num_docs)
    cols where each entry i,j is 1 if word i has label j and 0 otherwise
    :param label_list: dict where keys = labels and values = index
                        that maps to that label
    :param filename: name of the file that maps doc numbers to 
                    labels
    '''
    binary_label_vectors = {label: [0.0 for i in range(num_docs)] for label in indexed_labels.keys()}
    j = 0
    with open(filename, "r") as f:
        for line in f:
            terms = line.split()
            if terms[0][0:4] == "test":
                continue
            for elem in terms[1:]:
                binary_label_vectors[elem][j] = 1.0
            j += 1
    return binary_label_vectors                              


if __name__ == '__main__':
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\TextClassifiers\\training"
    stop_words = set(stopwords.words('english'))
    valid_words = fs.most_useful_features("cdmscores.txt")
    # valid_words = pp.get_valid_words(dir_path, stop_words)
    number_labels_training, number_labels_test = pp.add_labels_to_samples("cats2.txt")
    prior_probs = pp.compute_prior_probabilities(number_labels_training)
    indexed_labels = {label: i for i, label in enumerate(prior_probs.keys())}

    # build_new_cats_file("cats.txt")
    feature_matrix = construct_feature_matrix(dir_path, valid_words)
    print(feature_matrix.shape)
    # Each column in the feature matrix corresponds to an individual document

    weights = {label: np.asarray([np.random.uniform(0,0.005) for i in range(len(valid_words.keys()))]) for label in prior_probs.keys()}

    num_docs = len([name for name in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, name))])
    binary_label_vectors = construct_binary_label_vectors(indexed_labels, "cats2.txt", num_docs)

    trained_weights = logistic_regression(weights, feature_matrix, binary_label_vectors, 0.01, 0.01)

    successes = 0
    j = 0
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\TextClassifiers\\test"
    for file in os.listdir(dir_path):
        num = int(file[0:len(file) - 4])
        filepath = dir_path + "\\" + file
        vector = pp.vectorize_text(valid_words, filepath)
        frequencies = {word: 0 for word in valid_words}
        freq = Counter(vector)
        for word in freq.keys():
            frequencies[word] = freq[word]
        counts = np.asarray([v for v in frequencies.values()])
        probs = []
        for label, w in trained_weights.items():
            prob = np.dot(w, counts)
            probs.append([label, prob[0]])
        probs = sorted(probs, key=lambda t: t[1], reverse=True)
        just_scores = np.array([y for x,y in probs])
        just_scores = softmax(just_scores)
        for i in range(len(just_scores)):
            probs[i][1] = just_scores[i]
        s,e,b = pp.accuracy_model(num, number_labels_test, probs)
        successes += s
        j += 1
        # First run, 86.07% accuracy
    print(successes, j)
            


