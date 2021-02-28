import numpy as np
import math
import operator

# Constants
BIAS = 1
ITERATIONS = 50


class Preceptron:
    def __init__(self):
        self.weights = np.array([])
        self.learning_rate = 0.1

    '''
    Training function:
    Takes the X,y values and trains the perceptron on these values.
    Loops until the number of mispredictions (wrong counter) is zero (converges).
    At each point, find the prediction and if there is a misprediction, then
    adjust the weights; weights = weights + (y*X).
    '''

    def train_perceptron(self, X_train, y_train):
        number_of_weights = X_train.shape[1]
        number_of_instances = X_train.shape[0]
        self.weights = np.array([0.0] * (number_of_weights + 1))
        wrong_counter = 0
        for j in range(ITERATIONS):
            wrong_counter = 0
            for i in range(number_of_instances):
                training_instance_data = X_train[i].A[0].T.data
                training_instance_data = np.append(training_instance_data, BIAS)

                y_predict = np.dot(training_instance_data, self.weights)
                if y_predict * y_train[i] <= 0:
                    wrong_counter += 1
                    self.weights += (y_train[i] * training_instance_data)
            if wrong_counter == 0:
                break
        return self.weights

    '''
    Prediction function:
    returns the value of the weight vector by the data point.
    '''

    def predict(self, instance):
        training_instance_data = instance.A[0].T.data
        training_instance_data = np.append(training_instance_data, BIAS)
        result = np.dot(training_instance_data, self.weights)
        if result <= 0:
            return -1
        return 1

    def score(self, X_test, y_test):
        number_of_test_instances = X_test.shape[0]
        correct_prediction_counter = 0
        for i in range(number_of_test_instances):
            y_prediction = self.predict(X_test)
            if y_prediction == y_test[i]:
                correct_prediction_counter += 1
        accuracy = correct_prediction_counter / number_of_test_instances
        return accuracy

