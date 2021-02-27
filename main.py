import numpy as np
from sklearn.datasets import load_svmlight_file
from classifiers.knn import KNN
from classifiers.decision_tree import DecisionTree
from classifiers.preceptron import Preceptron
from classifiers.multiclassPerceptron import MulticlassPreceptron
import pandas as pd
from joblib import Memory
import matplotlib.pyplot as plt

mem = Memory("./mycache")


# @mem.cache
def load_data_set(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]


def split_data_set(X, y, test_size, subsample_size):
    if subsample_size > X.shape[0]:
        subsample_size = X.shape[0]

    shuffler = np.random.permutation(len(y))
    X_shuffled = X[shuffler]
    y_shuffled = y[shuffler]

    training_size = int((1 - test_size) * subsample_size)
    X_train = X_shuffled[:training_size]
    X_test = X_shuffled[training_size:subsample_size]
    y_train = y_shuffled[:training_size]
    y_test = y_shuffled[training_size:subsample_size]
    return X_train, X_test, y_train, y_test


def main():
    data_set_subsample_size = 300
    # X, y = load_data_set("datasets/multiclass/iris.scale")
    X, y = load_data_set("datasets/binary/a4a")

    X_train, X_test, y_train, y_test = split_data_set(X, y,
                                                      test_size = 0.3,
                                                      subsample_size = data_set_subsample_size)

    # KNN STUFF here
    # knn = KNN(neighbors=5)
    # knn.fit(X_train, y_train)
    # print(knn.score(X_test, y_test))

    # Perceptron stuff below
    perc = Preceptron()
    weights = perc.train_perceptron(X_train, y_train)
    score = perc.score(X_test, y_test)
    print(weights)
    print(score)

    # possible_classes = [int(element) for element in set(y_train)]
    # m_perceptron = MulticlassPreceptron(possible_classes)
    # weights = m_perceptron.train_perceptron(X_train, y_train)
    # predictions = m_perceptron.predict(X_test)
    # score = m_perceptron.score(X_test, y_test)
    # print(score)

    # fig, ax = plt.subplots(len(weights), sharex=True)
    # features = np.arange(X.shape[1])
    # print(features)
    # i = 0
    # for key, weight_vector in weights.items():
    #     len_of_features = len(features)
    #     # Slicing to remove the last weight (associated with the bias)
    #     absolute_value_weights = [abs(w) for w in weight_vector[:len_of_features]]
    #     ax[i].bar(features, absolute_value_weights)
    #     ax[i].set_title('Importance of weights for Class: ' + str(key))
    #     ax[i].set(xlabel='Feature Number', ylabel='Weight value')
    #     i += 1
    # fig.tight_layout(pad=1.0)
    # plt.show()


if __name__ == '__main__':
    main()
