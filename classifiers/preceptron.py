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
        self.weights = np.random.rand(number_of_weights)  * 2 + 1
        counter = 0
        wrong_counter = 11
        while wrong_counter > 0:
            wrong_counter = 0
            for i in range(number_of_instances):
                y_prime = self.predict([X_train[i]])
                difference = (y_train[i] - y_prime)
                if difference != 0:
                    wrong_counter += 1
                else:
                    continue
                delta_weights = self.learning_rate * difference * X_train[i].A[0].data
                self.weights += delta_weights
            print('wrong counter: '+str(wrong_counter))
        print(self.weights)

    def predict(self, instances):
        predictions = np.array([])
        sum = 0
        # THIS IS FOR ONE INSTANCE PREDICTION ONLY TODO
        for instance in instances:
            instance_data_array = instance.A[0]
            summation = np.dot(instance_data_array.T.data,self.weights)

        if summation > 0:
            return 2
        return 1



    # TODO: Compare the below with the old above code
    def train_perceptron(self, X_train, y_train):
        number_of_weights = X_train.shape[1]
        number_of_instances = X_train.shape[0]
        self.weights = np.random.rand(number_of_weights)  * 2 + 1
        self.weights = np.array([0] * number_of_instances)
        wrong_counter = 0
        while wrong_counter > 0:
            wrong_counter = 0
            for i in range(number_of_instances):
                y_predict = self.predict([X_train[i]])
                if y_train[i] != y_predict:
                    wrong_counter += 1
                    delta_weights = self.learning_rate * y_train[i] * X_train[i].A[0].data
                    self.weights += delta_weights

    def predict(self, instance):
        instance_data_array = instance.A[0]
        prediction = np.dot(instance_data_array.T.data,self.weights)
        if prediction > 0:
            return 1
        return -1
