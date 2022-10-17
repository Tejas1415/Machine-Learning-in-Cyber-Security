# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 20:26:29 2022

@author: Tejas

Manasa Assignment
"""

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression


# ### Reading-in the Iris data

s = os.path.join('https://archive.ics.uci.edu', 'ml',
                 'machine-learning-databases', 'iris','iris.data')
s = s.replace("\\","/");
print('URL:', s)
df = pd.read_csv(s,header=None,encoding='utf-8')



# select setosa and versicolor for binary classification
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1],
            color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1],
            color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')


# plt.savefig('images/02_06.png', dpi=300)
plt.show()


from matplotlib.colors import ListedColormap


def plot_decision_regions(X, y, classifier, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot class examples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], 
                    y=X[y == cl, 1],
                    alpha=0.8, 
                    c=colors[idx],
                    marker=markers[idx], 
                    label=cl, 
                    edgecolor='black')


class LogisticRegressionGD(object):
    """Logistic Regression Classifier using gradient descent.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    loss_ : list
      Logistic loss function value in each epoch.

    """
    def __init__(self, eta=0.05, n_iter=100, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
          Training vectors, where n_examples is the number of examples and
          n_features is the number of features.
        y : array-like, shape = [n_examples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.loss_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            output = self.activation(net_input)
            #print(output, '\n', i)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            
            # compute the logistic `loss` 
            loss = -y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output)))
            self.loss_.append(loss)
        return self
    
    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        """Compute logistic sigmoid activation"""
        return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, 0)
        # equivalent to:
        # return np.where(self.activation(self.net_input(X)) >= 0.5, 1, 0)




##### Question 1.1
X1 = X[0:3]
y1 = y[0:3]
initial_weights = [0.1, -0.2, 0.1]
eta = 0.1

input_ = np.dot(X1, initial_weights[1:]) + initial_weights[0]
print(input_)

## sigmoid activation
output_ = 1. / (1. + np.exp(-np.clip(input_, -250, 250)))
print(output_)


## Errors
err_ = y1 - output_
print(err_)

initial_weights[1:] += eta * X1.T.dot(err_)
initial_weights[0]  += eta * err_.sum()

### Weights after first iteration
print(initial_weights)








#### Question 1.2
from sklearn.preprocessing import PolynomialFeatures
X2 = PolynomialFeatures(2).fit_transform(X)
X3 = PolynomialFeatures(3).fit_transform(X)







#### Question 1.3
modl_X = LogisticRegressionGD(eta=0.1, n_iter=1050, random_state=1).fit(X, y)
loss_X = modl_X.loss_
weights_X = modl_X.w_

plot_decision_regions(X=X, y=y, classifier=modl_X)


modl_X2 = LogisticRegressionGD(eta=0.1, n_iter=1050, random_state=1).fit(X2, y)
loss_X2 = modl_X2.loss_
weights_X2 = modl_X2.w_


modl_X3 = LogisticRegressionGD(eta=0.1, n_iter=1020, random_state=1).fit(X3, y)
loss_X3 = modl_X3.loss_
weights_X3 = modl_X3.w_


print("Loss for X = ", loss_X[0])
print("\nLoss for X2 = ", loss_X2[0])
print("\nLoss for X3 = ", loss_X3[0])

'''
Markdown cell

Here, we can clearly observe the loss function value decrease as we increase the value of d,
i.e,
loss_X[0] > loss_X2[0] > loss_X3[0]
'''





######### Question 2:
    
'''
Markdown cell

Assuming data X is of the form [a, b], the second-degreee quadratic polynomial 
features for X would be [1, a, b, a^2, ab, b^2]. 

Loss funtion is generally a lower value if the number of correct predictions are higher. 
That is if the number of TruePositives (TP) and FalseNegatives(FN) are higher. If the model has more 
False Positives and True Negatives, the loss function value gets higher. 

Inorder to get more right predictions, the new columns added must provide better correlations to the outcome y, in a way the 
data is more separable. 

The additional features generated in a quadratic feature X2 scenario are (a^2, ab, b^2).

For example, if the model f is a linear classifier. (a,b) features in X are correlated to help developing a linear classifier, then
a^2, b^2 also will be correlated. However feature 'ab' might add more variance into the model reducing the models capacity to make
more accurate predictions leading to a larger loss function. 

If the model under consideration is a decision tree based algorithm, then feature 'ab' might be given very less weight (importance),
and hence it might not make a difference in prediction capabilities if 'ab' is not correlated and hence the loss function might be 
equal to X.

if the newly generated feature 'ab' reduces bias and is a more correlated with the outcomes leading to a better classifier, then the 
resultant loss function f1 (with data X2) will be lesser than f (with data X). 

In conclusion, the loss function difference depends on the model being used and the inherent nature of the data. Depending 
on what kind of variance the quadratic features add in the classification/regression problem, the loss function could be same, higher or lower 
compared to the loss function using raw X data.

'''



########### Question 3: 

    
### Q3.1
## creating our own linearly separable data 
import matplotlib.pyplot as plt
from sklearn import datasets

X, y = datasets.make_blobs(n_samples=1000, centers=2, n_features=2, center_box=(0, 20))
plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'g^')
plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bs')
plt.show()


### Q3.2
modl_X_new = LogisticRegressionGD(eta=0.1, n_iter=1050, random_state=1).fit(X, y)


### weights for the above data are as below
print("Optimal Weights for the above Logistic Neuron Model are =",  modl_X_new.w_)


### Q3.3
plot_decision_regions(X=X, y=y, classifier=modl_X_new)

'''
Markdown cell: 
    Here we are clearly able to recognize that the hyperplane separating the 2 classes is almost touching one class
    and leaves out a large margin on the other class. 
    
    Now lets try using sklearn logistic regression package to see if adjusting 'C' solves this issue.
'''


#### Q3.4
from sklearn.linear_model import LogisticRegression
modl_sklearn_LR = LogisticRegression(C = 3).fit(X, y)
plot_decision_regions(X=X, y=y, classifier=modl_sklearn_LR)


'''
Here we can observe that adjusting C=3 creates a middle hyperplane thats almost equidistant from both classes. 
'''





########### Question 4:

### Reading IRIS data again
# select setosa and versicolor for binary classification
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[:100, [0, 2]].values


## Question 4.1
from sklearn.svm import LinearSVC

modl_linearSVC = LinearSVC(C = 2).fit(X, y)
plot_decision_regions(X=X, y=y, classifier=modl_linearSVC)




### Question 4.2
## weights
modl_linearSVC.coef_


## Intercept
modl_linearSVC.intercept_


'''
Markdown cell: 

Hence, the hyperplane equation is as follows:
    y = -0.644x1 + 1.3857x2 - 0.221
'''

## Finding 2-norm of the above weights

intercept_SVC = modl_linearSVC.intercept_
weights_SVC = modl_linearSVC.coef_
norm2 = np.linalg.norm(weights_SVC, ord = 2)
print("2-norm s = ", norm2)




### Question 4.3
weights_SVC1 = weights_SVC/norm2
intercept_SVC1 = intercept_SVC/norm2
intercept_SVC1 = intercept_SVC1[0]



### Question 4.4
## apply w1x1T - b for datapoints in first column, and w2x2T-b for all elements in second column

X_1 = list(X[:, 0]) ## first column  
X_2 = list(X[:, 1]) ## other column

gamma1 = [weights_SVC1[0][0]*x - intercept_SVC1 for x in X_1]
print(gamma1)

print(min(gamma1))

gamma2 = [weights_SVC1[0][1]*x - intercept_SVC1 for x in X_2]
print(gamma2)

print(min(gamma2))


### least absolute value
gamma1_least_abs = abs(min(gamma1))
gamma1_least_abs


gamma2_least_abs = abs(min(gamma2))
gamma2_least_abs






######### Question 5

##### Question 5.1
## 2-norm for every element in X
X_norm2 = []
for i in range(X.shape[0]):
    X_norm2.append(np.linalg.norm(X[i,:], ord = 2))


R = max(X_norm2)
print(R)


#### Question 5.2
maxErrors = R/(gamma1_least_abs**2)


#### Question 5.3
## reusing code from assignment 2 for perceptron

class Perceptron(object):
     """Perceptron classifier.
     Parameters
     ------------
     eta : float
     Learning rate (between 0.0 and 1.0)
     n_iter : int
     Passes over the training dataset.
     random_state : int
     Random number generator seed for random weight
     initialization.
     Attributes
     -----------
     w_ : 1d-array
     Weights after fitting.
     errors_ : list
     Number of misclassifications (updates) in each epoch.
     """
     def __init__(self, eta=0.01, n_iter=50, random_state=1):
         self.eta = eta
         self.n_iter = n_iter # Attribute for iterations
         self.weights = [] # Attribute for weights
         self.random_state = random_state
         
     def fit(self, X, y):
        """Fit training data.
        Parameters
        ----------
        X : {array-like}, shape = [n_examples, n_features]
        Training vectors, where n_examples is the number of examples and
        n_features is the number of features.
        y : array-like, shape = [n_examples]
        Target values.
        Returns
        -------
        self : object
        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []
       
       
        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.weights.append(self.w_) # storing weights at each iteration to main
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0: # stops when no more iterations are necessary
                break
            # my do-nothing code
            IK = 2020
            # my do-nothing code
       
        return self
    
            
     def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

     def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.net_input(X) >= 0.0, 1, -1)
        
      
modl_perceptron = Perceptron(eta=0.1, n_iter=1000).fit(X,y)       
        

### Errors
modl_perceptron.errors_    
        
### Max error   
maxError_perceptron = max(modl_perceptron.errors_ )
        

'''
Markdown cell

In Our case, we can see that the MaxError obtained was approximately 1. Whereas using a perceptron, max errors possible 
before convergence are 3. Hence, logisticRegressionGD has lower possibilities of exceeding errors before convergence.
'''
 


        
        
        
        
        
    
    
    
    
    
    
    
