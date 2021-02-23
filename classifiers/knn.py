import numpy as np
import math
import operator

class KNN:
    def __init__(self, neighbors=5, distance_method='euclidean'):
        self.k_neighbors = neighbors
        self.X_train = None
        self.y_train = None
        self.number_of_features = 0
        self.distance_metric = distance_method


    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.number_of_features = self.X_train.shape[1]

    def predict(self, instances):
        predictions = np.array([])
        for instance in instances:
            top_k_neighbors = self.find_k_nearest_instances(instance)
            prediction = self.vote_on_instance(top_k_neighbors)
            predictions = np.append(predictions, prediction)
        return predictions

    def find_k_nearest_instances(self, new_instance):
        distance_array = self.initialize_distance_array(new_instance)
        sorted_distance_array = sorted(distance_array, key=lambda i: i['distance'])
        top_k_neighbors = sorted_distance_array[0:self.k_neighbors]
        return top_k_neighbors

    def vote_on_instance(self, top_k_neighbors):
        classes = {}
        for neighbor in top_k_neighbors:
            classification = int( self.y_train[neighbor['id']])
            if str(classification) in classes:
                classes[str(classification)] += 1
            else:
                classes[str(classification)] = 1
        return int(max(classes.items(), key=operator.itemgetter(1))[0])

    def find_distance(self, row, new_instance):
        sum = 0
        for i in range(self.number_of_features):
            if self.distance_metric == 'euclidean':
                sum += (new_instance[0,i] - row[0,i]) ** 2
            else:
                sum += math.abs(new_instance[0,i] - row[0,i])
        if self.distance_metric == 'euclidean':
            return math.sqrt(sum)
        else:
            return sum

    def initialize_distance_array(self, new_instance):
        length = self.X_train.shape[0]
        distance_array = []
        for i in range(length):
            element = {'distance': self.find_distance(self.X_train[i], new_instance), 'id': i}
            distance_array.append(element)
        return distance_array


    def score(self, X_test, y_test):
        test_length = X_test.shape[0]
        correct = 0
        # pred_instances = X_test[0]
        predictions = self.predict(X_test)
        for i in range(len(predictions)):
            if int(predictions[i]) == int(y_test[i]):
                correct += 1
        return correct / test_length
