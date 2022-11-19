# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 19:27:08 2022

@author: Tejas
"""


from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split


X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
y = y.astype(int)
X = ((X / 255.) - .5) * 2
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=41, stratify=y)




### Question 1.1

print("Number of datapoints in training dataset = ", X_train.shape[0])

print("Number of datapoints in test dataset = ", X_test.shape[0])


### Question 1.2
print("Number of attributes in X = ", X_train.shape[1])


'''
Markdown cell
We observe that X_train has 60,000 rows and X_test has 10,000 rows. 
Each row represents 784 pixels in a 28x28 handwritten digit picture as attributes. 

Hence, we have 60,000 unique image samples in training and 10,000 unique image samples in test dataset.
'''


### Question 1.3

## Let us check the unique number of elements in y_train
print("Total number of unique labels in y_train = ", y_train.nunique())
print("\nThose Unique labels are as follows: ", y_train.unique())


### Question 1.4
'''
Markdown cell

We know that the size of one image sample in MNIST database is 28x28

28x28 = 784

The pixels in the image are reshaped into one long row (1x784) in our given dataset. 
X_train and X_test have 784 columns (attributes) which correspond to the 28x28 pixels in each image.
'''


### Question 1.5
''' 
Markdown cell

Initially in the fetched dataset from fetch_openml, they are grayscale images 
having pixel values ranging from 0-255. 
Where 0 generally is the darkest color gradient in the image and 255 is white color representation. 


Gradient based optimizations always work most stable when the data is scalled/ normalized. 
Hence, in order to scale the data between a close range [-1, 1] the below line is used.

X = ((X / 255.) - .5) * 2

For example if a pixel is 0,
(0/255 - 0.5) * 2 = -1

And if a pixel is 255,
(255/255 - 0.5) * 2 = 1

Hence all pixels in range [0, 255] will now be scaled between [-1, 1] which helps the modeling the data for better predictions.
'''





##### Question 2

## Let us fit the pca with X_train, and use that to transform both x_train and x_test datasets
from sklearn.decomposition import PCA

pca = PCA(n_components=2).fit(X_train)
X_train_transformed = pca.transform(X_train)
X_test_transformed = pca.transform(X_test)


X_train_transformed


X_test_transformed





#### Question 3.1:
    
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                           metric='minkowski', metric_params=None, n_jobs=None)
knn = knn.fit(X_train, y_train)



#### Question 3.2:

y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)


from sklearn.metrics import accuracy_score

print("Training Accuracy = ", accuracy_score(y_train, y_pred_train))
print("\nTesting Accuracy = ", accuracy_score(y_test, y_pred_test))


### Question 3.3
'''
If we replaced every unique label with another random label consistently in both train and test datasets, 
For example, 
If all images of hand written digit 4 were to be replaced with a random label 91 (consistantly for all 4, replace 91).
And similarly for all labels, 
then the accuracy would not change. Because KNN classifier segregates the data samples based on how similar they are,
and checks if they have same label or not. As long as they are consistantly replaced with random variable the accuracy would not change. 


However, if every test sample is mapped with a completely random label. Then the classifier accuracy would drastically reduce.
KNN classifer would still segregate similar datapoints into clusters, and when checked their labels would be very different, hence the accuracy 
would be very low (Mostly accuracy = 0, unless there are accidental true positives while randomly assigning the label)  

'''




### Question 3.4 

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2,
                           metric='minkowski', metric_params=None, n_jobs=None)
knn = knn.fit(X_train, y_train)


y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)


from sklearn.metrics import accuracy_score

print("Training Accuracy = ", accuracy_score(y_train, y_pred_train))
print("\nTesting Accuracy = ", accuracy_score(y_test, y_pred_test))
























