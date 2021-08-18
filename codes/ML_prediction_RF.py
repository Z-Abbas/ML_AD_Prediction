#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 13:49:39 2021

@author: zeeshan
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle


######
# DEG_DMP Integrated
######

######################
training_acc_arr = []
test_acc_arr = []
roc_auc_arr = []

index=0
for index in range(5):
    # Using the own splitted Train / Test data
    train = pd.read_csv("K_Fold_Data_64/Integrated/"+str(index)+'_train_DEG_DMP.csv', delimiter = ",")
    train_X = train.iloc[:,2:-1]
    train_y = train.iloc[:,-1]
    
    test = pd.read_csv("K_Fold_Data_64/Integrated/"+str(index)+'_test_DEG_DMP.csv', delimiter = ",")
    test_X = test.iloc[:,2:-1]
    test_y = test.iloc[:,-1]
    ######################
    
    rdf = RandomForestClassifier(criterion='gini', n_estimators=40, n_jobs=-1, random_state=0, max_depth=4)
    rdf.fit(train_X,train_y) #oob_score=True,
    
    # *****save the model to disk *****
    filename = "K_Fold_Data_64/Integrated/"+str(index)+'_fold_finalized_model_RF.sav'
    pickle.dump(rdf, open(filename, 'wb'))
    
    predicted = rdf.predict(test_X)
    confusion_matrix(test_y,predicted)
    
    test_acc = accuracy_score(test_y, predicted) # Testing Accuracy
    training_acc = accuracy_score(train_y, rdf.predict(train_X)) # Training Accuracy
    roc_auc = metrics.roc_auc_score(test_y.ravel(), predicted.ravel())
    
    # Appending in list to take average
    training_acc_arr.append(training_acc)
    test_acc_arr.append(test_acc)
    roc_auc_arr.append(roc_auc)
    
    print(str(index) + "-fold training accuracy:" + "\t" + str(training_acc))
    print(str(index) + "-fold test accuracy:" + "\t" + str(test_acc))
    print(str(index) + "-fold test roc_auc:" + "\t" + str(roc_auc))
    print('\n')

    
    
    index+=1

# Averaging    
mean_tr_acc = np.mean(training_acc_arr)
mean_test_acc = np.mean(test_acc_arr)
mean_roc_auc = np.mean(roc_auc_arr)
print('Mean Training Accuracy:'+ "\t" + str(mean_tr_acc))
print('Mean Testing Accuracy:'+ "\t" + str(mean_test_acc))
print('Mean ROC AUC:'+ "\t" + str(mean_roc_auc))


#%%
# Predicting DEG and DMP seperately without integration
######
# DEG
######
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle

training_acc_arr = []
test_acc_arr = []
roc_auc_arr = []

index=0
for index in range(5):

    train = pd.read_csv("K_Fold_Data_64/DEG/"+str(index)+'_train_DEG.csv', delimiter = ",")
    train_X = train.iloc[:,0:-2]
    train_y = train.iloc[:,-1]
    
    test = pd.read_csv("K_Fold_Data_64/DEG/"+str(index)+'_test_DEG.csv', delimiter = ",")
    test_X = test.iloc[:,0:-2]
    test_y = test.iloc[:,-1]
    ######################
    
    rdf = RandomForestClassifier(criterion='gini', n_estimators=10, n_jobs=-1, random_state=0, max_depth=5)
    rdf.fit(train_X,train_y)
    
    # *****save the model to disk *****
    filename = "K_Fold_Data_64/DEG/"+str(index)+'_fold_finalized_model_RF.sav'
    pickle.dump(rdf, open(filename, 'wb'))
    
    predicted = rdf.predict(test_X)
    confusion_matrix(test_y,predicted)
    
    test_acc = accuracy_score(test_y, predicted) # Testing Accuracy
    training_acc = accuracy_score(train_y, rdf.predict(train_X)) # Training Accuracy
    roc_auc = metrics.roc_auc_score(test_y.ravel(), predicted.ravel())
    
    # Appending in list to take average
    training_acc_arr.append(training_acc)
    test_acc_arr.append(test_acc)
    roc_auc_arr.append(roc_auc)
    
    print(str(index) + "-fold training accuracy:" + "\t" + str(training_acc))
    print(str(index) + "-fold test accuracy:" + "\t" + str(test_acc))
    print(str(index) + "-fold test roc_auc:" + "\t" + str(roc_auc))
    print('\n')
 

    index+=1

# Averaging    
mean_tr_acc = np.mean(training_acc_arr)
mean_test_acc = np.mean(test_acc_arr)
mean_roc_auc = np.mean(roc_auc_arr)
print('Mean Training Accuracy:'+ "\t" + str(mean_tr_acc))
print('Mean Testing Accuracy:'+ "\t" + str(mean_test_acc))
print('Mean ROC AUC:'+ "\t" + str(mean_roc_auc))


#%%
# Predicting DEG and DMP seperately without integration
######
# DMP
######
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pickle

training_acc_arr = []
test_acc_arr = []
roc_auc_arr = []

index=0
for index in range(5):
    train = pd.read_csv("K_Fold_Data_64/DMP/"+str(index)+'_train_DMP.csv', delimiter = ",")
    train_X = train.iloc[:,0:-2]
    train_y = train.iloc[:,-1]
    
    test = pd.read_csv("K_Fold_Data_64/DMP/"+str(index)+'_test_DMP.csv', delimiter = ",")
    test_X = test.iloc[:,0:-2]
    test_y = test.iloc[:,-1]
    ######################
	
    rdf = RandomForestClassifier(criterion='gini', n_estimators=10, n_jobs=-1, random_state=0, max_depth=5)
    rdf.fit(train_X,train_y)
    
    # *****save the model to disk *****
    filename = "K_Fold_Data_64/DMP/"+str(index)+'_fold_finalized_model_RF.sav'
    pickle.dump(rdf, open(filename, 'wb'))

    predicted = rdf.predict(test_X)
    confusion_matrix(test_y,predicted)
    
    test_acc = accuracy_score(test_y, predicted)
    training_acc = accuracy_score(train_y, rdf.predict(train_X))
    roc_auc = metrics.roc_auc_score(test_y.ravel(), predicted.ravel())
    
    # Appending in list to take average
    training_acc_arr.append(training_acc)
    test_acc_arr.append(test_acc)
    roc_auc_arr.append(roc_auc)
    
        
    print(str(index) + "-fold training accuracy:" + "\t" + str(training_acc))
    print(str(index) + "-fold test accuracy:" + "\t" + str(test_acc))
    print(str(index) + "-fold test roc_auc:" + "\t" + str(roc_auc))
    print('\n')
    
    
    index+=1
 
# Averaging    
mean_tr_acc = np.mean(training_acc_arr)
mean_test_acc = np.mean(test_acc_arr)
mean_roc_auc = np.mean(roc_auc_arr)
print('Mean Training Accuracy:'+ "\t" + str(mean_tr_acc))
print('Mean Testing Accuracy:'+ "\t" + str(mean_test_acc))
print('Mean ROC AUC:'+ "\t" + str(mean_roc_auc))

    
