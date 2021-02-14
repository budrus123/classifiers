import numpy as np
from sklearn.datasets import load_svmlight_file
import math
import operator
from sklearn.model_selection import train_test_split


class KNN:
    def __init__(self, neighbors=5, distance_method='euclidean'):
        self.k_neighbors = neighbors
        self.X_train = None
        self.y_train = None
        self.distance_method = distance_method

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

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
        top_k_neighbors = sorted_distance_array[0:k_neighbors]
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
        row_data = row[0].data
        data_length = len(row_data)
        sum = 0
        for i in range(data_length):
            if self.distance_metric == 'euclidean':
                sum += (new_instance[i] - row_data[i]) ** 2
            else:
                sum += math.abs(new_instance[i] - row_data[i])
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


dataset_subsample = 760
def load_dataset_file():
    data = load_svmlight_file("datasets/binary/diabetes")
    return data[0], data[1]

def main():
    X, y = load_dataset_file()
    y = y[0:dataset_subsample]
    X = X[0:dataset_subsample, ]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        shuffle=True)

    diabetes_instance = np.array([3.000000, 162.000000, 52.000000, 38.000000, 0.000000, 37.200001, 0.652000, 24.000000])
    iris_instance = np.array([0.388889, -0.166667, 0.525424, 0.666667])

    instances = np.array([diabetes_instance])
    knn = KNN()
    knn.fit(X_train, y_train)
    print(instances)
    predictions = knn.predict(instances)
    print(predictions)
    # top_k_neighbors = find_k_nearest_instances(X_train, diabetes_instance)
    # prediction = vote_on_instance(top_k_neighbors, y_train)
    # # print(prediction)
    # accuracy = find_accuracy(X_train, y_train, X_test, y_test)
    # print('Accuracy is: ' + str(accuracy * 100))


if __name__ == '__main__':
    main()
