import numpy as np
import math
import operator

# Constants
BIAS = 1
ITERATIONS = 10000


class MulticlassPreceptron:
    def __init__(self, possible_outcomes):
        self.learning_rate = 0.1
        self.X_train = None
        self.y_train = None
        self.classes = set(possible_outcomes)
        self.number_of_features = 0
        self.number_of_classes = len(self.classes)
        self.weights = {}

    def initialize_weights(self):
        for data_class in self.classes:
            self.weights[str(data_class)] = np.array([0.0] * (self.number_of_features + 1))

    def find_closest_class(self, training_instance_data):
        max_prediction = 0
        max_prediction_class = 0
        for perceptron_class in self.classes:
            prediction = np.dot(training_instance_data, self.weights[str(perceptron_class)])
            if prediction >= max_prediction:
                max_prediction = prediction
                max_prediction_class = perceptron_class

        return max_prediction_class

    def train_perceptron(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.number_of_features = X_train.shape[1]
        number_of_training_instances = X_train.shape[0]
        self.initialize_weights()
        print(self.weights)
        # exit(0)
        for i in range(ITERATIONS):
            for j in range(number_of_training_instances):
                training_instance_data = X_train[j].A[0].T.data
                training_instance_data = np.append(training_instance_data, BIAS)
                y_pred = self.find_closest_class(training_instance_data)  # TODO: Actually find the prediction
                if y_pred != y_train[j]:
                    self.weights[str(int(y_train[j]))] += training_instance_data
                    self.weights[str(int(y_pred))] -= training_instance_data
        print(self.weights)

    def predict(self, test_instances):
        number_of_test_instances = test_instances.shape[0]
        predictions = np.array([])
        for j in range(number_of_test_instances):
            training_instance_data = test_instances[j].A[0].T.data
            training_instance_data = np.append(training_instance_data, BIAS)
            y_pred = self.find_closest_class(training_instance_data)  # TODO: Actually find the prediction
            predictions = np.append(predictions, y_pred)
        return predictions

    def score(self, X_test, y_test):
        number_of_test_instances = X_test.shape[0]
        y_predictions = self.predict(X_test)
        correct_prediction_counter = 0
        for i in range(number_of_test_instances):
            if y_predictions[i] == y_test[i]:
                correct_prediction_counter += 1

        accuracy = correct_prediction_counter / number_of_test_instances
        return accuracy

    '''
    Training function:
    Takes the X,y values and trains the perceptron on these values.
    Loops until the number of mispredictions (wrong counter) is zero (converges).
    At each point, find the prediction and if there is a misprediction, then
    adjust the weights; weights = weights + (y*X).
    '''
    # def train_perceptron(self, X_train, y_train):
    #     number_of_weights = X_train.shape[1]
    #     number_of_instances = X_train.shape[0]
    #     self.weights = np.array([0.0] * (number_of_weights+1))
    #     wrong_counter = 0
    #     while True:
    #         wrong_counter = 0
    #         for i in range(number_of_instances):
    #             training_instance_data = X_train[i].A[0].T.data
    #             training_instance_data = np.append(training_instance_data, BIAS)
    #             y_predict = np.dot(training_instance_data, self.weights)
    #             difference = (y_train[i] - y_predict)
    #             print(difference)
    #             if difference <= 0:
    #                 wrong_counter += 1
    #                 delta_weights = (difference * X_train[i].A[0].data)
    #                 self.weights += delta_weights
    #         print(wrong_counter)
    #         if wrong_counter == 0:
    #             break
    #     print(self.weights)
    # '''
    # Prediction function:
    # returns the value of the weight vector by the data point.
    # '''
    # def predict(self, instance):
    #     instance_data_array = instance.A[0]
    #     result = np.dot(instance_data_array.T.data,self.weights)
    #     if result > 1.5:
    #         return 2
    #     return 1
