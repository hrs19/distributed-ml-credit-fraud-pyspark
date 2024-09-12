

import numpy as np
import pandas as pd
import math
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from time import time
from sklearn.metrics import r2_score, accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix

# Input the test and Train datasets
start_time = time()
raw_df = pd.read_csv('/scratch/tarasia.dev/Project/creditcard.csv')
end_time = time()
print("Time Taken to read original data: {:1.5f}".format(end_time-start_time))

start_time = time()
raw_smote_df = pd.read_csv('/scratch/tarasia.dev/Project/oversample.csv')
end_time = time()
print("Time Taken to read oversampled data: {:1.5f}".format(end_time-start_time))

raw_df.columns = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]
raw_smote_df.columns = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount","Class"]

X_actual = np.array(raw_df.drop(["Class"], axis=1))  
y_actual = np.array(raw_df["Class"])            

X_oversample = np.array(raw_smote_df.drop(["Class"], axis=1))  
y_oversample = np.array(raw_smote_df["Class"])            

#Divide training set into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_actual, y_actual, test_size=0.3, random_state = 42)

# Preprocessing X in a way it produces better results.
"""
Feature Scaling - Standard Scaling technique 
(Because this works best for SVM's, logistic regression, and other classification algorithms)
 
1. For every feature (column), find mean.
2. Subtract mean from each value in the column.
3. Divide the answer with the standard deviation of initial column.

"""
start = time()
X_train = StandardScaler().fit_transform(X_train)
X_test = StandardScaler().fit_transform(X_test)
end = time()
print("Time Taken to standardize actual data: {:1.5f}".format(end_time-start_time))


########################## PreProcessing Complete #############################

from sklearn.ensemble import ExtraTreesClassifier 

# getiing 100% and 76.5% train n test acc - (n = 500)
rf = RandomForestClassifier(n_estimators = 100,
                            criterion = "gini",
                           n_jobs = 1,
                           oob_score = True,
                           max_features = "sqrt",
                           max_depth = 10,
                           min_samples_leaf = 4,
                           random_state = 42)

# Calculate Model Training time
start_time = time()
rf.fit(X_train, y_train)
end_time = time()
print("Time Taken to fit actual data to model: {:1.5f}".format(end_time-start_time))
# Output 14.2747280 secs

rf.get_params() ## To check the automatic hyperparameters chosen by Rf.

## Building an algorithm is fast in Random forests but predicting results consumes considerable amount of time.
start_time = time()
predicted = rf.predict(X_test)
end_time = time()
print("Time Taken to test actual data: {:1.5f}".format(end_time-start_time))
# output 0.547907352 secs


print ("Training Accuracy of actual data:", accuracy_score(y_train, rf.predict(X_train)))
print ("Testing Accuracy of actual data:",  accuracy_score(y_test, rf.predict(X_test)))
    
false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, rf.predict(X_test))
roc_auc = auc(false_positive_rate, true_positive_rate)
print ("AUC Score: ", roc_auc)   ## AUC score is better 
print ("False Positive Rate of actual data:", false_positive_rate[1])
print ("True Positive Rate of actual data:", true_positive_rate[1])
print("Confusion Matrix of actual data \n",confusion_matrix(y_test,predicted))
