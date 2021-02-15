import numpy as np
from sklearn.datasets import load_svmlight_file
import math
import operator
from sklearn.model_selection import train_test_split


class Preceptron:
    def __init__(self):
        self.weights = np.array([])
        self.learning_rate = 0.1

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        number_of_weights = X_train.shape[1]
        number_of_instances = X_train.shape[0]
        print(X_train.shape)
        self.weights = np.random.rand(number_of_weights)  * 2 + 1
        print(self.weights)
        counter = 0
        wrong_counter = 11
        while wrong_counter > 0:
            wrong_counter = 0
            for i in range(number_of_instances):
                # TODO REMOVE THIS, To handle missing data, TODO REMOVE THIS
                if len(X_train[i].data) != len(self.weights):
                    continue
                y_prime = self.predict([X_train[i].data])
                difference = (y_train[i] - y_prime)
                if difference != 0:
                    wrong_counter += 1
                else:
                    continue
                delta_weights = self.learning_rate * difference * X_train[i].data
                # print(delta_weights)
                self.weights += delta_weights
                # print(difference)
            print('wrong counter: '+str(wrong_counter))
        print(self.weights)

    def predict(self, instances):
        predictions = np.array([])
        sum = 0

        for instance in instances:
            summation = np.dot(instance.T.data,self.weights)
        # print(summation)
        if summation > 0:
            return 2
        return 1

    def find_k_nearest_instances(self, new_instance):
        print()

    def score(self, X_test, y_test):
        return 0
