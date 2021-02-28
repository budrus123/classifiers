import numpy as np

# Constants
BIAS = 1
ITERATIONS = 50


class MulticlassPreceptron:
    ''''''

    '''
    Initializer function to takes the possible classes for the 
    data set.
    '''

    def __init__(self, possible_classes):
        self.learning_rate = 0.1
        self.X_train = None
        self.y_train = None
        self.classes = set(possible_classes)
        self.number_of_features = 0
        self.number_of_classes = len(self.classes)
        self.weights = {}

    '''
    Function initialize the weights dictionary with zeros with 
    the same length as the feature vector + 1 for the BIAS.
    
    Each class will have its own weights in the dictionary.
    '''

    def initialize_weights(self):
        for data_class in self.classes:
            self.weights[str(data_class)] = np.array([0.0] * (self.number_of_features + 1))

    '''
    3. Compute the predicted outcome for each single instance in the data set. 
    The outcome is computed as follows:
        - For every class in the total number of classes,
        compute the product of that class weight vector, with the 
        data instance.
        - Return the class that causes the biggest activation, meaning
        the biggest product among all the different classes.
        '''

    def find_closest_class(self, training_instance_data):
        max_prediction = 0
        max_prediction_class = 0
        for perceptron_class in self.classes:
            prediction = np.dot(training_instance_data, self.weights[str(perceptron_class)])
            if prediction >= max_prediction:
                max_prediction = prediction
                max_prediction_class = perceptron_class

        return max_prediction_class

    '''
    Main Algorithm for the Multi-class Perceptron is as follows:

    1. Initialize the weights dictionary with p weight vectors,
    where p is the number of distinct classes or outcomes in the
    data set.

    2. Train the weight vector on the data set by a predefined
    number of iterations.

    3. In each iteration, compute the predicted outcome for each single
    instance in the data set. The outcome is computed as follows:
        - For every weight vector in the weights dictionary,
        compute the product of that weight vector, with the 
        data instance.
        - Return the class that causes the biggest activation, meaning
        the biggest product among all the different classes.

    4. If the prediction class is the same as the expected class, then 
    do nothing.

    5. If the prediction class is different from the expected class, then:
        - Add the feature vector to the weights of the expected class.
        - Subtract the feature vector from the weights of the predicted (wrong) class.

    '''

    def train_perceptron(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.number_of_features = X_train.shape[1]
        number_of_training_instances = X_train.shape[0]
        self.initialize_weights()
        for i in range(ITERATIONS):
            for j in range(number_of_training_instances):
                training_instance_data = X_train[j].A[0].T.data
                training_instance_data = np.append(training_instance_data, BIAS)
                y_pred = self.find_closest_class(training_instance_data)
                if y_pred != y_train[j]:
                    self.weights[str(int(y_train[j]))] += training_instance_data
                    self.weights[str(int(y_pred))] -= training_instance_data
        return self.weights

    '''
    Function that takes a list of test instances, and returns an
    array of predictions for those instances.
    '''

    def predict(self, test_instances):
        number_of_test_instances = test_instances.shape[0]
        predictions = np.array([])
        for j in range(number_of_test_instances):
            training_instance_data = test_instances[j].A[0].T.data
            training_instance_data = np.append(training_instance_data, BIAS)
            y_pred = self.find_closest_class(training_instance_data)
            predictions = np.append(predictions, y_pred)
        return predictions

    '''
    Function that returns the score (accuracy) of the Multi-class Perceptron.

    It takes a testing feature array and a testing outcome array.
    The score is calculated by returning the number of correct 
    classifications out of the total number of testing instances.
    '''

    def score(self, X_test, y_test):
        number_of_test_instances = X_test.shape[0]
        y_predictions = self.predict(X_test)
        correct_prediction_counter = 0
        for i in range(number_of_test_instances):
            if y_predictions[i] == y_test[i]:
                correct_prediction_counter += 1

        accuracy = correct_prediction_counter / number_of_test_instances
        return accuracy
