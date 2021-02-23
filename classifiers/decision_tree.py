import numpy as np
from sklearn.datasets import load_svmlight_file
import math
import operator
from sklearn.model_selection import train_test_split
from scipy.stats import entropy

class DecisionTree:
    def __init__(self, neighbors=5, distance_method='euclidean'):
        self.k_neighbors = neighbors
        self.X_train = None
        self.y_train = None
        self.number_of_features = 0
        self.distance_metric = distance_method

    def fit(self, X_train, y_train):
        print(X_train)
        print(y_train)

    def predict(self, instances):
        print()

    def score(self, X_test, y_test):
        print()
