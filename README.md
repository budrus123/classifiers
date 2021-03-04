# K-NN and Perceptron Classifiers

CONTENTS OF THIS FILE
---------------------

 * [Introduction](#intro)
 * [K-NN Classifier](#knn)
 * [Perceptron Classifier](#perceptron)

INTRODUCTION<a name="intro"></a> 
------------
In this project, two classifiers were created. The first classifier was a K-NN classifier and the second one was a simple Perceptron.
Both were created from scratch and both used two different datasets.
The first dataset was the a4a dataset, which is a binary outcome dataset that can have an outcome of either +1 or -1.
The other was the famous multiclass Iris dataset which can have an outcome of 1, 2, or 3.


K-NN Classifier<a name="knn"></a>    
------------
K-Nearest Neighbors (or K-NN) is a simple classification algorithm that creates a decision boundary by exploring the K-nearest instances to the instance one wishes to classify. The classification depends on the classes (or votes) that the K-nearest instances have. For example, if K was set to 3, then this means that when the classifier wishes to classify an instance A, then this classification algorithm will look at the 3 nearest instances to A using a distance function (either Euclidean or Manhattan distance). It will then see which class is reflected the most in the 3 nearest instances.
In this project, a K-NN class was created to allow the instantiation of a K-NN classifier object that takes the number of neighbors as well as the distance method that will be used (either Euclidean or Manhattan distance). After that, the classifier needs to be fit with the training data. This basically just sets the training data (which will be used in finding the nearest neighbors) to the classifier object. After that, we can use the classifier to predict the outcome for a set of instances and a list of predictions will be returned. An example of this usage can be shown below.

```python
knn = KNN(neighbors=5, distance_method='euclidean')
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)
print(knn.score(X_test, y_test))
```

The steps of how the K-NN classifier works and how the predictions are returned are described below.
1.	The K-NN classifier object is created with the needed number of neighbors and the wanted distance method.
2.	The training data is set to the classifier (the fitting of the data to the classifier)
3.	The classifier is used to predict the outcome of a set of instances. For each instance, the outcome is predicted as follows:
    1. An array of objects is created to calculate the distance between the current instance and all the instances in the training data. Each object has the instance id (row number) and the distance to the instance we wish to calculate.
    2. The array of distances is sorted depending on the distance metric (in ascending fashion).
    3. The top K neighbors (closest neighbors) are selected.
    4. The most occurring class (class that appears the most) in the top K neighbors is the class that represents the instance we wish to predict.
4.	To calculate the accuracy of the classifier, a set of testing data can be fed into the classifier (testing feature data and testing target data). In the score function, predictions are created for the testing feature data which is then compared to the actual target data. The accuracy is calculated as follows:


Perceptron Classifier<a name="perceptron"></a>  
------------


Usage
------------

```sh
$ python main.py <image_name.jpg>
```

