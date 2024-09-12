from sklearn.linear_model import SGDClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time

start_time = time.time()
# Load data 
data_df = pd.read_csv("creditcard1.csv")
read_time = time.time() - start_time

# Split data
train_df = data_df.sample(frac=0.7, random_state=1234)
test_df = data_df.drop(train_df.index)
X_train = train_df.iloc[:, :-1].to_numpy()
y_train = train_df.iloc[:, -1].to_numpy()
X_test = test_df.iloc[:, :-1].to_numpy()
y_test = test_df.iloc[:, -1].to_numpy()

# Scale the data 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the SVM model
clf = SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=10000, tol=1e-3, random_state=1234)
start_time = time.time()
clf.fit(X_train_scaled, y_train)
train_time = time.time() - start_time


y_pred = clf.predict(X_test_scaled)


precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1_score = f1_score(y_test, y_pred)

print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1-score: {:.4f}".format(f1_score))
print("Training time: {:.4f} seconds".format(train_time))
print("Read data time: {:.4f} seconds".format(read_time))
