import numpy as np
from sklearn.datasets import load_svmlight_file
import math
import operator
from sklearn.model_selection import train_test_split
from scipy.stats import entropy
from collections import Counter


class DecisionTree:
    def __init__(self, X_train, y_train, value_type):
        self.X_train = None
        self.y_train = None
        self.number_of_features = 0
        self.number_of_classes = len(set(y_train))
        self.X_train = X_train.toarray()
        self.y_train = y_train
        self.classes = set(y_train)
        self.value_type = value_type
        self.feature_count = np.diff(X_train.tocsc().indptr)

    def fit(self, X_train, y_train):
        self.number_of_features = self.X_train.shape[1]
        # print(X_train.shape)
        #
        # # number of times te feature was 1
        #
        # print(len(self.feature_count))
        # print(self.feature_count)
        dataset_entropy = self.entropy_of_dataset(self.X_train, self.y_train)
        self.relative_entropy_of_feature(self.X_train, self.y_train, 0)
        i_g0 = self.information_gain_of_feature(self.X_train, self.y_train, 0)
        print(i_g0)

    def predict(self, instances):
        print()

    def score(self, X_test, y_test):
        print()

    def entropy_of_dataset(self, X, y):
        number_of_classes = len(set(y))
        classes = set(y)
        probs = []
        for outcome in classes:
            count_of_outcome = np.count_nonzero(y == outcome)
            probs.append(count_of_outcome)
        entropy_of_ds = entropy(probs, base=2)
        return entropy_of_ds

    def entropy_of_outlook_for_feature(self, X, y, feature_index, value):

        subset = y[X[:, feature_index] == value]
        # print(y[X[:, feature_index] == value])
        counter = Counter(subset)
        # print(counter)
        if len(counter) == 1:
            return 0
        else:
            pairs = []
            for key, item in counter.items():
                pairs.append(item)
        entr = entropy(pairs, base=2)
        return entr

    def relative_entropy_of_feature(self, X, y, feature_index):
        total_number_of_rows = X.shape[0]
        feature_values = {}
        if self.value_type == 'binary':
            feature_values['1'] = {'value': 1, 'count': self.feature_count[feature_index]}
            feature_values['0'] = {'value': 0, 'count': X.shape[0] - self.feature_count[feature_index]}

        entropy_sum = 0
        # print(feature_values)
        for key, item in feature_values.items():
            probability = item['count'] / total_number_of_rows
            # print(probability)
            outlook_of_target = 0
            entropy_of_outlook_for_feature = self.entropy_of_outlook_for_feature(X, y, feature_index, item['value'])
            entropy_sum += probability * entropy_of_outlook_for_feature
        return entropy_sum

    def information_gain_of_feature(self, X, y, feature_index):
        entropy_of_ds = self.entropy_of_dataset(X, y)
        relative_entropy_feature = self.relative_entropy_of_feature(X, y, feature_index)
        return entropy_of_ds - relative_entropy_feature
