# -*- coding: utf-8 -*-
"""
Machine learning Example

Created on Thu May  7 10:38:11 2020

@author: gsapi
"""


#################### Load Libraries ##########################################

# pandas is an open source data analysis and manipulation tool, built on top 
# of the Python programming language
# sklearn is a free software machine learning library for the Python 
# programming language.
from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

# load 
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# load test algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

############# Load Dataset and Separate for Analysis #########################
# Load dataset
url = "https://raw.githubusercontent.com/gsapijaszko1/Datasets/master/iris.csv"
names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Category']
dataset = read_csv(url, names=names)

# Separate dataset for analysis
data = dataset.values
samples = data[:,0:4] #collect first 4 columns for analysis - samples
cat = data[:,4] #Collect category
samples_train, samples_validation, cat_train, cat_validation = train_test_split(samples, cat, test_size=0.20, random_state=1)

#################### Build and Evaluate Models ###############################
#WWhich algorithms would be a good fit for this dataset?
#Models will be compared to each other and the most accurate will be tested.

# Six different models will be tested and accuracy estimations calculated:

# Logistic Regression (LR)
# Linear Discriminant Analysis (LDA)
# K-Nearest Neighbors (KNN).
# Classification and Regression Trees (CART).
# Gaussian Naive Bayes (NB).
# Support Vector Machines (SVM).

# This is a good mixture of simple linear (LR and LDA), 
# and nonlinear (KNN, CART, NB and SVM) algorithms.

# Build and evaluate our models:
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

# Evaluate each model one at a time
print('\nBuild and Evaluate Models:')
for name, model in models:
 	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True) #train on 9, test on 1 for kfold
 	cv_results = cross_val_score(model, samples_train, cat_train, cv=kfold, scoring='accuracy') #This is on 80% train, 20% test
 	print('%s: %0.2f%%' % (name, cv_results.mean()*100))
print()
##############Select Best Model and Make Predictions##########################
     
# Make predictions on validation dataset
model = SVC(gamma='auto') #support vector machine - supervised learning method used for classification
model.fit(samples_train, cat_train)
predictions = model.predict(samples_validation)
print('Make Predictions with Model Chosen:')
# Evaluate predictions
wrongCount = 0
i = 0
for val, pred in zip(cat_validation, predictions):
    i = i + 1
    if (val == pred):
        print(str(i) + ") actual: " + str(val) + "\t predicted: " + str(pred))
    else:
        print(str(i) + ") actual: " + str(val) + "\t predicted: " + str(pred) + "\t INCORRECT") 
        wrongCount = wrongCount + 1
print('\nSVM Accuracy: ','%0.2f%%' % (accuracy_score(cat_validation, predictions)*100), ' - ', wrongCount,'/', len(cat_validation), ' INCORRECT')
