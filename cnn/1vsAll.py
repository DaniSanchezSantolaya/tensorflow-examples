

import numpy as np
import pylab as plt
import os
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score


idx = np.arange(50000)
np.random.shuffle(idx)
idx = idx[:25000]

layers = ['fc1','fc2','flatten']
features_path = './features/task1/reg'
#layers = ['L2-norm']
#features_path = './features/task2'

#Load labels train
labels_file_train = os.path.join('./features', 'labels_train.npy')
labels_train = np.load(labels_file_train)
print labels_train.shape
labels_train = np.argmax(labels_train, axis=1)
labels_train = labels_train[idx]
#Load labels test
labels_file_test = os.path.join('./features', 'labels_test.npy')
labels_test = np.load(labels_file_test)
print labels_test.shape
labels_test = np.argmax(labels_test, axis=1)


for layer in layers:
    print layer
    #Load all train features images    
    features_file_train_1 = os.path.join(features_path, 'features_' + str(layer) + '_train_0.npy')
    features_file_train_2 = os.path.join(features_path, 'features_' + str(layer) + '_train_1.npy')
    features_file_train_3 = os.path.join(features_path, 'features_' + str(layer) + '_train_2.npy')
    features_file_train_4 = os.path.join(features_path, 'features_' + str(layer) + '_train_3.npy')
    features_file_train_5 = os.path.join(features_path, 'features_' + str(layer) + '_train_4.npy')
    batch_1 = np.load(features_file_train_1)
    batch_2 = np.load(features_file_train_2)
    batch_3 = np.load(features_file_train_3)
    batch_4 = np.load(features_file_train_4)
    batch_5 = np.load(features_file_train_5)
    X_train = np.concatenate((batch_1, batch_2, batch_3, batch_4, batch_5))
    X_train = X_train[idx]
    print 'X_train shape: ' + str(X_train.shape)
    #Load test features images
    features_file_test = os.path.join(features_path, 'features_' + str(layer) + '_test.npy')  
    X_test = np.load(features_file_test)
    print 'X_test.shape: ' + str(X_test.shape)
    #Train OneVsRestClassifier
    oneVsRest = OneVsRestClassifier(LinearSVC(random_state=0))
    print 'train...'
    oneVsRest.fit(X_train, labels_train)
    #Predict and evaluate
    y_pred = oneVsRest.predict(X_test)
    accuracy = accuracy_score(labels_test, y_pred)
    print 'Accuracy in layer ' + str(layer) + ': ' + str(accuracy)
    print '----------'
    




