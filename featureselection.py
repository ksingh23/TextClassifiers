from matplotlib import pyplot as plt
import numpy as np
from collections import OrderedDict


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


def most_useful_features(filename):
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
    most_useful = {w: True for w, c in cdm[i:]}
    print(len(most_useful.keys()))
    return OrderedDict(sorted(most_useful.items(), key=lambda t: t[0]))


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
    most_useful_features("cdmscores.txt")