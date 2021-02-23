import numpy as np
from sklearn.datasets import load_svmlight_file
import math
import operator
from sklearn.model_selection import train_test_split
from scipy.stats import entropy


class DecisionTree:
    def __init__(self, X_train, y_train):
        self.X_train = None
        self.y_train = None
        self.number_of_features = 0
        self.number_of_classes = len(set(y_train))
        self.X_train = X_train
        self.y_train = y_train
        self.classes = set(y_train)

    def fit(self, X_train, y_train):
        self.number_of_features = self.X_train.shape[1]
        print(X_train.shape)

        # number of times te feature was 1
        feature_count = np.diff(X_train.tocsc().indptr)
        print(len(feature_count))
        dataset_entropy = self.entropy_of_dataset(self.X_train, self.y_train)
        print(dataset_entropy)

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
