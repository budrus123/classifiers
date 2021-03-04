import numpy as np
from sklearn.datasets import load_svmlight_file
# classifiers is a directory in my project
# to import the different classifiers
# (check github repo for more info)
from classifiers.KNN import KNN
from classifiers.Preceptron import Preceptron
from classifiers.MulticlassPerceptron import MulticlassPreceptron
import matplotlib.pyplot as plt

'''
Function to load the data set and return the feature
vector and the output vector.
'''

def load_data_set(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]


'''
Function that splits the data into training and testing
data and sub-samples the data to a certain size (randomly).
'''

def split_data_set(X, y, test_size, subsample_size):
    if subsample_size > X.shape[0]:
        subsample_size = X.shape[0]

    random_permutation = np.random.permutation(len(y))
    X_randomized = X[random_permutation]
    y_randomized = y[random_permutation]

    training_size = int((1 - test_size) * subsample_size)
    X_train = X_randomized[:training_size]
    X_test = X_randomized[training_size:subsample_size]
    y_train = y_randomized[:training_size]
    y_test = y_randomized[training_size:subsample_size]
    return X_train, X_test, y_train, y_test


'''
Function to visualize the weights.
'''

def visualize_weights(weights, number_of_features):
    if isinstance(weights, dict):
        visualize_weight_object(weights, number_of_features)
    else:
        visualize_weight_array(weights, number_of_features)


'''
Function to visualize the weights when the weights are 
a dictionary (weights for different classes).
'''

def visualize_weight_object(weights, number_of_features):
    fig, ax = plt.subplots(len(weights), sharex=True)
    features = np.arange(number_of_features)
    i = 0
    for key, weight_vector in weights.items():
        # Slicing to remove the last weight (associated with the bias)
        absolute_value_weights = [abs(w) for w in weight_vector[:number_of_features]]
        ax[i].bar(features, absolute_value_weights)
        ax[i].set_title('Importance of weights for Class: ' + str(key))
        ax[i].set(xlabel='Feature Number', ylabel='Weight value')
        i += 1
    fig.tight_layout(pad=1.0)
    plt.show()


'''
Function to visualize the weights when the weights are 
an array (weights for binary +1/-1 data set).
'''

def visualize_weight_array(weights, number_of_features):
    fig, ax = plt.subplots()
    features = np.arange(number_of_features)
    absolute_value_weights = [abs(w) for w in weights[:number_of_features]]
    ax.bar(features, absolute_value_weights)
    ax.set_title('Importance of weights')
    ax.set(xlabel='Feature Number', ylabel='Weight value')
    fig.tight_layout(pad=1.0)
    plt.show()


'''
Function to plot the different values of the accuracy
along with different values of K.
'''

def plot_accuracy_for_k_values(k_values, accuracy):
    fig, ax = plt.subplots()
    ax.plot(k_values, accuracy)
    ax.set_title('Relationship between K and Accuracy of KNN')
    ax.set(xlabel='K - Values', ylabel='Accuracy')
    plt.show()


def main():
    data_set_subsample_size = 1000
    X, y = load_data_set("datasets/multiclass/iris.scale")
    # X, y = load_data_set("datasets/binary/a4a")
    X_train, X_test, y_train, y_test = split_data_set(X, y,
                                                      test_size=0.3,
                                                      subsample_size=data_set_subsample_size)

    # KNN STUFF here
    knn = KNN(neighbors=5, distance_method='euclidean')
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    score = knn.score(X_test, y_test)
    print("----------- K-NN Classifier ----------- ")
    print("Score of K-nn is: " + str(score))

    # Perceptron Code here
    print("----------- Perceptron ----------- ")
    possible_classes = [int(element) for element in set(y_train)]
    if len(possible_classes) == 2:
        perceptron = Preceptron()
    else:
        perceptron = MulticlassPreceptron(possible_classes)

    weights = perceptron.train_perceptron(X_train, y_train)
    score = perceptron.score(X_test, y_test)
    print("Score of perceptron is: " + str(score))
    print("Visualizing weight importance")
    visualize_weights(weights, X_train.shape[1])


if __name__ == '__main__':
    main()
