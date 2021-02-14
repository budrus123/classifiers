import numpy as np
from sklearn.datasets import load_svmlight_file
import math
import operator
from sklearn.model_selection import train_test_split


class Preceptron:
    def __init__(self):
        print()

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, instances):
        predictions = np.array([])
        for instance in instances:
            print()
        return predictions

    def find_k_nearest_instances(self, new_instance):
        print()

    def score(self, X_test, y_test):
        return 0
