from matplotlib import pyplot as plt
import numpy as np
from nltk.corpus import stopwords
import preprocessing as pp
from collections import OrderedDict, defaultdict
import os
import naivebayes as nb


def polynomial(a, x, b, c):
    return a * (x ** b) + c


def logarithmic(a, x, b):
    return a * np.log(x) + b


def log_derivative(a, x):
    return a/x


def log_second_derivative(x):
    return -91.4755/(x**2)


def graph_polynomial(a, x, b, c, color):
    y = polynomial(a, x, b, c)
    plt.plot(x, y, color=color)


def graph_logarithm(a, x, b, color):
    y = logarithmic(a, x, b)
    plt.plot(x, y, color=color)


def graph_log_derivative(a, x, color):
    y = log_derivative(a, x)
    plt.plot(x, y, color=color)


def compute_mse(function, x, params, y_pred):
    a, b, c = params[0], params[1], params[2]
    return ((function(a, x, b, c) - y_pred)**2).mean()


def most_useful_features_by_cdm(filename):
    cdm_scores = []
    words = []
    cdm = []
    with open("cdmscores.txt", "r") as f:
        i = 0
        for line in f:
            terms = line.split()
            word = terms[0][:len(terms[0]) - 1]
            words.append(word)
            score = float(terms[1])
            cdm.append([word, score])
            cdm_scores.append(score)
            i += 1
    cdm_scores.reverse()
    cdm_scores = np.asarray(cdm_scores)
    # cdm_scores += 388.35237183375955
    cdm_scores += 389
    indices = np.asarray([j for j in range(1, i + 1)])
    log_coeff = np.polyfit(np.log2(indices), cdm_scores, 1)
    a, b = log_coeff[0], log_coeff[1]
    a /= np.log(2)
    # Curve = 63.406log2(x) - 293.35138 = 91.4755ln(x) - 293.35138
    # d/dx (Curve) = 91.4755/x
    desired_delta = 1.0 * (10 ** -6)
    # print(desired_delta)
    y = log_derivative(a, indices)
    i = 1
    # print(y[i], y[i - 1], np.absolute(y[i] - y[i - 1]))
    while np.absolute(y[i] - y[i - 1]) >= desired_delta:
        i += 1
    # len(valid_words) = 23799
    most_useful = {w: True for w, c in cdm[18799:]}
    print(len(most_useful.keys()))
    return OrderedDict(sorted(most_useful.items(), key=lambda t: t[0]))


def chi_squared_test(dir_path, valid_words_labels, number_labels, prior_probs, total_valid_words):
    # D = num_docs
    # P = num docs with class C containing term T
    # Q = num docs not with class C containing term T
    # N = number of documents not with class C not containing term T
    # M = number of documents with class C not containing term T
    d = len([file for file in os.listdir(dir_path)])
    df = pp.get_df(dir_path, number_labels, prior_probs.keys(), total_valid_words)
    parameters = pp.get_parameters(dir_path, total_valid_words, number_labels, prior_probs.keys())
    idf = parameters[2]
    valid_words_by_label = {label: defaultdict(bool) for label in prior_probs.keys()}
    for label, vector in valid_words_labels.items():
        for word, score in vector.items():
            if word in df[label]:
                p = df[label][word]
            else:
                p = 0
            total_docs_with_word = (d/np.exp(idf[word] - 1)) - 1
            n = d - prior_probs[label] - (total_docs_with_word - df[label][word])
            m = prior_probs[label] - df[label][word]
            q = total_docs_with_word - df[label][word]
            denom = (p+m)*(q+n)*(p+q)*(m+n)
            print(word, p, m, q, n)
            # Chi Square statistic for p-value < 0.05 is 3.841 (1 DF)
            # Higher chi squared statistic means that there is no relationship
            # between the term and the label
            # So we want to pick the ones with which we fail to reject the null
            # hypothesis, i.e any scores < 3.841
            chi_squared_stat = (d*(p*n-m*q)**2)/denom
            # print(label, word, chi_squared_stat)
            if chi_squared_stat <= 5.02:
                valid_words_by_label[label][word] = True
        valid_words_taken = valid_words_by_label[label]
        valid_words_by_label[label] = OrderedDict(valid_words_taken.items(), key=lambda t: t[0])
    return valid_words_by_label


if __name__ == '__main__':
    # cdm scores decrease minimally after highest 9564 words
    # In the future, it might be worthwhile to chart the
    # value of desired_delta vs the number of useful words
    # and desired_delta vs model training time
    '''
    plt.plot(indices, cdm_scores)
    graph_logarithm(a, indices, b, 'red')
    graph_log_derivative(a, indices, 'purple')
    plt.show()
    '''
    dir_path = "C:\\Users\\ksing\\OneDrive\\Documents\\TextClassifiers\\MiniTrainingSet"
    stop_words = set(stopwords.words('english'))
    number_labels_training, number_labels_test = pp.add_labels_to_samples("cats1.txt")
    prior_probs = nb.compute_prior_probabilities(number_labels_training)
    valid_words, total_valid_words = pp.get_valid_words(dir_path, stop_words, prior_probs.keys(), number_labels_training)
    valid_words_by_label = chi_squared_test(dir_path, valid_words, number_labels_training, prior_probs, total_valid_words)
    for label, vector in valid_words_by_label.items():
        print(label, len(vector.keys()), len(valid_words[label]))


