import numpy as np
from sklearn.datasets import load_svmlight_file
import math
import operator
from sklearn.model_selection import train_test_split


k_neighbors = 7
dataset_subsample = 760

def load_dataset_file():
    data = load_svmlight_file("datasets/binary/diabetes")
    return data[0], data[1]

def find_distance(row, new_instance):
    row_data = row[0].data
    data_length = len(row_data)
    squared_sum = 0
    for i in range(data_length):
        squared_sum += (new_instance[i] - row_data[i]) ** 2
    return math.sqrt(squared_sum)

def initialize_distance_array(X,new_instance):
    length = X.shape[0]
    distance_array = []
    for i in range(length):
        element = {'distance': find_distance(X[i], new_instance), 'id': i}
        distance_array.append(element)
    return distance_array

def find_k_nearest_instances(X, new_instance):
    distance_array = initialize_distance_array(X,new_instance)
    sorted_distance_array = sorted(distance_array, key = lambda i: i['distance'])
    top_k_neighbors = sorted_distance_array[0:k_neighbors]
    return top_k_neighbors

def vote_on_instance(top_k_neighbors, y):
    classes = {}
    for neighbor in top_k_neighbors:
        classification = int(y[neighbor['id']])
        if str(classification) in classes:
            classes[str(classification)] += 1
        else:
            classes[str(classification)] = 1
    return max(classes.items(), key=operator.itemgetter(1))[0]

def find_accuracy(X_train, y_train, X_test, y_test):
    test_length = X_test.shape[0]
    correct = 0
    for i in range(test_length):
        instance = X_test[i].data
        top_k_neighbors = find_k_nearest_instances(X_train, instance)
        prediction = vote_on_instance(top_k_neighbors, y_train)
        if int(prediction) == int(y_test[i]):
            correct += 1
    return correct / test_length

def main():
    X, y = load_dataset_file()
    y = y[0:dataset_subsample]
    X = X[0:dataset_subsample,]
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        shuffle=True)

    diabetes_instance = np.array([3.000000, 162.000000, 52.000000, 38.000000, 0.000000, 37.200001, 0.652000, 24.000000])
    iris_instance = np.array([0.388889, -0.166667, 0.525424, 0.666667 ])
    top_k_neighbors = find_k_nearest_instances(X_train, diabetes_instance)
    prediction = vote_on_instance(top_k_neighbors, y_train)
    # print(prediction)
    accuracy = find_accuracy(X_train, y_train, X_test, y_test)
    print('Accuracy is: ' + str(accuracy * 100))

if __name__ == '__main__':
    main()