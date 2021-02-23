import numpy as np
import math
import operator


class KNN:
    ''''''
    '''
    Initializer function to take the number of neighbors and
    the distance metric.
    '''

    def __init__(self, neighbors=5, distance_method='euclidean'):
        self.k_neighbors = neighbors
        self.X_train = None
        self.y_train = None
        self.number_of_features = 0
        self.distance_metric = distance_method

    '''
    Fit function for KNN. Function that stores the training feature
    data and the training target data.
    '''

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.number_of_features = self.X_train.shape[1]

    '''
    Function to predict a set of instances.
    
    Steps for the function:
    1. Fnd the K nearest neighbors
    2. Use nearest neighbors to vote on the classification of instance
    3. Append to prediction array
    4. Return prediction array.
    '''

    def predict(self, instances):
        predictions = np.array([])
        for instance in instances:
            top_k_neighbors = self.find_k_nearest_instances(instance)
            prediction = self.vote_on_instance(top_k_neighbors)
            predictions = np.append(predictions, prediction)
        return predictions

    '''
    Function to find the K-nearest neighbors to an instance.
    
    Steps for function:
    1. Initialize distance array (array of objects, each of which has the 
    instance row number and the distance to our new instance).
    2. Sort the array of distances.
    3. Find the K nearest neighbors (by slicing the array) and return them
    '''

    def find_k_nearest_instances(self, new_instance):
        distance_array = self.initialize_distance_array(new_instance)
        sorted_distance_array = sorted(distance_array, key=lambda i: i['distance'])
        top_k_neighbors = sorted_distance_array[0:self.k_neighbors]
        return top_k_neighbors

    '''
    Function to use top K neighbors to vote on outcome.
    
    Steps for function:
    1. Loop he neighbor objects and find their original classification
    in the target (outcome) column
    2. Add classification to object if not already added, if already there
    then increase it's count by one.
    3. Return the most (max) occurring key (class) in our classes object.
    '''

    def vote_on_instance(self, top_k_neighbors):
        classes = {}
        for neighbor in top_k_neighbors:
            classification = int(self.y_train[neighbor['id']])
            if str(classification) in classes:
                classes[str(classification)] += 1
            else:
                classes[str(classification)] = 1
        return int(max(classes.items(), key=operator.itemgetter(1))[0])

    '''
    Function to find the distance between a row and a new instance (to 
    be predicted).
    
    The function loops the features and depending on the distance 
    metric used in the KNN object creation, it adds the corresponding
    distance difference to the sum. If the metric is euclidean, then
    it returns the square root of the sum, otherwise just the sum.
    '''

    def find_distance(self, row, new_instance):
        sum = 0
        for i in range(self.number_of_features):
            if self.distance_metric == 'euclidean':
                sum += (new_instance[0, i] - row[0, i]) ** 2
            else:
                sum += math.abs(new_instance[0, i] - row[0, i])
        if self.distance_metric == 'euclidean':
            return math.sqrt(sum)
        else:
            return sum

    '''
    Function to initialize the distance array between the new instance to be 
    predicted and all our training instances.
    '''

    def initialize_distance_array(self, new_instance):
        length = self.X_train.shape[0]
        distance_array = []
        for i in range(length):
            element = {'distance': self.find_distance(self.X_train[i], new_instance), 'id': i}
            distance_array.append(element)
        return distance_array

    '''
    Function that returs the score of our KNN classifier.
    
    It takes a testing feature array and a testing outcome array.
    The score is calculated by returning the number of correct 
    classifications out of the totall number of testing instances.
    '''

    def score(self, X_test, y_test):
        test_length = X_test.shape[0]
        correct = 0
        predictions = self.predict(X_test)
        for i in range(len(predictions)):
            if int(predictions[i]) == int(y_test[i]):
                correct += 1
        return correct / test_length
