# K-NN and Perceptron Classifiers

CONTENTS OF THIS FILE
---------------------

 * [Introduction](#intro)
 * [K-NN Classifier](#knn)
 * [Perceptron Classifier](#perceptron)
 * [Usage](#usage)

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

1. The K-NN classifier object is created with the needed number of neighbors and the wanted distance method.
2. The training data is set to the classifier (the fitting of the data to the classifier)
3. The classifier is used to predict the outcome of a set of instances. For each instance, the outcome is predicted as follows:
    1. An array of objects is created to calculate the distance between the current instance and all the instances in the training data. Each object has the instance id (row number) and the distance to the instance we wish to calculate.
    2. The array of distances is sorted depending on the distance metric (in ascending fashion).
    3. The top K neighbors (closest neighbors) are selected.
    4. The most occurring class (class that appears the most) in the top K neighbors is the class that represents the instance we wish to predict.
4. To calculate the accuracy of the classifier, a set of testing data can be fed into the classifier (testing feature data and testing target data). In the score function, predictions are created for the testing feature data which is then compared to the actual target data.


Perceptron Classifier<a name="perceptron"></a>  
------------
The Perceptron is a classification technique that classifies 
linearly separable data. It works by taking an input feature 
that is then multiplied by a weight vector, and it outputs a 
class depending on the dot product between the two. If the product 
is larger than a threshold, then it will be classified as class A (+1), 
but if the product is less than or equal to the threshold then it 
will be classified as class B (-1) or the other way around. 
The essence of the perceptron is the weight vector, 
which involves a lengthy training process to be learned. 

To make it much easier and compare with zero instead of a threshold, 
a feature of value 1 can be added to the feature vector, 
along with a weight that can be added to the weight vector and 
that can be learned along with the other weights.

Note that this project had two different types of datasets, one was binary and the 
other was multi-class data. Two different perceptron classes were created to 
classify each type of data. The reason for this is 
that the original perceptron algorithm can classify data that 
belongs to only two classes only. To expand this algorithm further 
to be able to classify data that has more than two classes, 
a Multiclass Perceptron algorithm was used. 
The multiclass perceptron algorithm can also 
be used for binary classification data (since it has +1 and -1 classes)
but for the sake of clarification and accuracy, the project 
chooses the correct type of perceptron to create 
depending on the number of classes that the dataset has.

To decide which type of perceptron to create, the code looks 
at the target vector (the output of the training data) 
and sees how many unique items (different classes there is). 
If there are only two classes, a regular perceptron is created. 
If there are more than two, a multiclass perceptron is created. 
This code can be observed below:

```python
    possible_classes = [int(element) for element in set(y_train)]
    if len(possible_classes) == 2:
        perceptron = Preceptron()
    else:
        perceptron = MulticlassPreceptron(possible_classes)
```

### Binary Perceptron

This is the simple perceptron approach that was explained above. It involves two main operations which are explained below:
1. The training operation (learning the weights): Here, the main goal is to learn the correct weights that can be used for the classification process this is done by using the `train_perceptron` function. The basic steps here are.

    1. Setting the initial weights to a zero vector of length the same as the feature vector +1 for the BIAS.
    2. Looping through the training instances and computing the product of the instance feature vector by the weight vector.
    3. In the case there is a misprediction, the weight vector is updated by adding the instance feature vector multiplied by the output value to the old weight vector.
    4. In case of a correct prediction, no operation takes place.
    5. This keeps happening until we have exhausted all of our preset iterations constant, or we have gone through the entire dataset with zero wrong predictions (found a linear separator for the dataset).
2. The prediction Operation: After the weights have been learned and set in the classifier, the next step is to actually use the weights for predicting never-before-seen test instances. This can be done by using the `predict` function which takes a list of instances and returns a list of predictions for those instances. Predicting instances is really simple, it only depends on the product of the trained weights and the test instance. If the product is greater than zero, then it belongs to the +1 class, otherwise, it belongs to the -1 class. Simple as that.


### Multiclass Perceptron

The ordinary perceptron can only tell us if a certain data instance belongs to a set of two classes. This is done through the activation function (or threshold value) that is set for the classification. In our above binary perceptron, if the product of the weights and feature vectors exceeded zero, then we know it belonged to the +1 class. But what happens when we have 3 or more classes? We cannot use the simple perceptron to classify instances here. For the project, the Iris dataset has 3 different classes, and this approach was used for it.
In this case, a different approach needs to be followed. The approach that was selected for this project depends on having multiple weight vectors (not just one). The number of weight vectors corresponds to the number of different classes we have. For example, for the Iris dataset, we will have three different weight vectors, each for their respective class. Meaning we will have a weight vector for class 1, another for class 2, and one final one for class 3. The multiclass perceptron has the same two main operations that the regular perceptron has (training and predicting), but they function in a different way. The main differences of the two are below:
1. The training operation: In this function, the training data is used to learn a set of weight vectors (not just one) by doing the following:
    1. Setting the initial weight vectors to a zero.
    2. Looping through the training instances and decide which weight vector is the most appropriate for this data instance. How? This is done by seeing which weight vector causes the biggest activation (biggest dot product value between the weight vector and the data instance). Once we know which weight vector causes the biggest activation, then we decide that this instance belongs to that class (the class that corresponds to that weight vector, see the ` find_closest_class` function in the multiclass perceptron code).
    3. In the case there is a misprediction, which means that our instance really belongs to class A but was classified as class B, then two updates are required. First, the weight vector of A needs to be updated by adding the feature vector of the instance to it (to make it lean more towards the instance in the future). In addition, the weight vector for class B needs to be updated by subtracting the feature vector of the instance from it (to make it move further away from the instance in the future).
    4. This keeps happening until we have exhausted all of our preset iterations constant.
2. The prediction Operation: After the weights have been learned and set in the classifier, the next step is to actually use the weights for predicting never-before-seen test instances. This can be done by using the `predict` function which takes a list of instances and returns a list of predictions for those instances. Predicting instances uses the ` find_closest_class` function to find the weight vector that causes the biggest activation and returns the class that corresponds to that weight vector.

Usage<a name="usage"></a> 
------------
>The main python file has both the K-NN classifier and the Perceptron running
> for the Iris dataset. In addition, visualization for most important features
> is prodived. __Classifier type and dataset file will be added as parameters__
```sh
$ python main.py
```
---
- [ ] Add classifier type as a parameter
 
- [ ] Add dataset as a parameter
 
- [ ] Add sub-sampling size and test size as a parameter


