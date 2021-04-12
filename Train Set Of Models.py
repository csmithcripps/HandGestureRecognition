
# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.python.solutions import hands as mp_hands

import handRepresentation

gestureThreshold = 0.8
# Split-out validation dataset
names = ['Gesture', 
                    'Thumb0',
                    'Thumb1',
                    'Thumb2',
                    'Pointer0',
                    'Pointer1',
                    'Pointer2',
                    'Index0',
                    'Index1',
                    'Index2',
                    'Ring0',
                    'Ring1',
                    'Ring2',
                    'Pinky0',
                    'Pinky1',
                    'Pinky2']
                    
dataset = read_csv('./GestureData.csv', names=names)

array = dataset.values
X = array[:,1:17]
y = array[:,0]
X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# evaluate each model in turn
results = {}
names = []
for name, model in models:
	kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
	cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
	results.update( {name: [cv_results.mean(), cv_results.std()]})
	names.append(name)

for key in results:
  print(key, 'Mean: ' + str(results[key][0]))

