import numpy as np
import math
import operator

# Constants
BIAS = 1
ITERATIONS = 10000


class Preceptron:
    def __init__(self):
        self.weights = np.array([])
        self.learning_rate = 0.1

    # def fit(self, X_train, y_train):
    #     self.X_train = X_train
    #     self.y_train = y_train
    #     number_of_weights = X_train.shape[1]
    #     number_of_instances = X_train.shape[0]
    #     self.weights = np.random.rand(number_of_weights)  * 2 + 1
    #     counter = 0
    #     wrong_counter = 11
    #     while wrong_counter > 0:
    #         wrong_counter = 0
    #         for i in range(number_of_instances):
    #             y_prime = self.predict([X_train[i]])
    #             difference = (y_train[i] - y_prime)
    #             if difference != 0:
    #                 wrong_counter += 1
    #             else:
    #                 continue
    #             delta_weights = self.learning_rate * difference * X_train[i].A[0].data
    #             self.weights += delta_weights
    #         print('wrong counter: '+str(wrong_counter))
    #     print(self.weights)
    #
    # def predict(self, instances):
    #     predictions = np.array([])
    #     sum = 0
    #     # THIS IS FOR ONE INSTANCE PREDICTION ONLY TODO
    #     for instance in instances:
    #         instance_data_array = instance.A[0]
    #         summation = np.dot(instance_data_array.T.data,self.weights)
    #
    #     if summation > 0:
    #         return 2
    #     return 1



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
        print(self.weights)

    '''
    Prediction function:
    returns the value of the weight vector by the data point.
    '''

    def predict(self, instance):
        training_instance_data = instance.A[0].T.data
        training_instance_data = np.append(training_instance_data, BIAS)
        return np.dot(training_instance_data, self.weights)

    def score(self, X_test, y_test):
        number_of_test_instances = X_test.shape[0]
        correct_prediction_counter = 0
        for i in range(number_of_test_instances):
            y_prediction = self.predict(X_test)
            print(y_prediction)
        #     if y_predictions[i] == y_test[i]:
        #         correct_prediction_counter += 1
        #
        # accuracy = correct_prediction_counter / number_of_test_instances
        return 0

