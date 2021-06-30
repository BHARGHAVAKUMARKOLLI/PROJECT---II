# -*- coding: utf-8 -*-
"""
Created on Wed May 26 03:06:13 2021

@author: bharghava
"""

import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifier as KNC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import seaborn as sns
iris=pd.read_csv('C:/Users/bharghava/Downloads/iris (1).csv', header=None)
iris.head()

## Load data
iris = datasets.load_iris()
X = iris.data
y = iris.target

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Create adaboost classifer object
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=50,
                         learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = model.predict(X_test)
#
from sklearn import metrics
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

####SVC MODEL : SUPPORT VECTOR CLASSIFIER
from sklearn.svm import SVC
from sklearn import metrics
svc=SVC(probability=True, kernel='linear')
### CREATING A MODEL BY USING adaboost by USING SVC
# Create adaboost classifer object
abc_svc=AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
# Train Adaboost Classifer
model_svc = abc.fit(X_train, y_train)
#Predict the response for test dataset
y_pred = model.predict(X_test)
# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))




























