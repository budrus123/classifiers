import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split
from classifiers.knn import KNN
from classifiers.decision_tree import DecisionTree
from classifiers.preceptron import Preceptron
from classifiers.multiclassPerceptron import MulticlassPreceptron
import pandas as pd
from joblib import Memory

mem = Memory("./mycache")

dataset_subsample = 300


# @mem.cache
def load_dataset(file_name):
    data = load_svmlight_file(file_name)
    return data[0], data[1]


def main():
    X, y = load_dataset("datasets/multiclass/iris.scale")
    # X, y = load_dataset("datasets/binary/a4a")
    y = y[0:dataset_subsample]
    X = X[0:dataset_subsample, ]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Decision tree below

    # dt = DecisionTree(X, y, 'binary')
    # dt.fit(X, y)


    # diabetes_instance = np.array([3.000000, 162.000000, 52.000000, 38.000000, 0.000000, 37.200001, 0.652000, 24.000000])
    # iris_instance = np.array([0.388889, -0.166667, 0.525424, 0.666667])

    # instances = np.array([X_test[0],X_test[0]])
    # print(instances:
    # [0].data)
    # for i in range(1,10):

    # KNN STUFF here
    # knn = KNN(neighbors=5)
    # knn.fit(X_train, y_train)
    # print(knn.score(X_test, y_test))

    # Perceptron stuff below
    # print(X_train)
    # print(y_train)
    # perc = Preceptron()
    # perc.train_perceptron(X_train, y_train)
    # score = perc.score(X_test, y_test)

    m_perceptron = MulticlassPreceptron([1,2,3])
    m_perceptron.train_perceptron(X_train, y_train)
    predictions = m_perceptron.predict(X_test)
    score = m_perceptron.score(X_test, y_test)
    print(score)

    '''
        Examples
    --------
    To use joblib.Memory to cache the svmlight file::

        from joblib import Memory
        from .datasets import load_svmlight_file
        mem = Memory("./mycache")

        @mem.cache
        def get_data():
            data = load_svmlight_file("mysvmlightfile")
            return data[0], data[1]

        X, y = get_data()
        '''


if __name__ == '__main__':
    main()
