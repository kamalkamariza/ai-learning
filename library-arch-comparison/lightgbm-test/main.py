#Import scikit-learn dataset library
from sklearn import datasets

# Import train_test_split function
from sklearn.model_selection import train_test_split

#Import lightgbm model
'''
    brew install open-mpi
    brew install lightgbm
    pip3 install lightgbm
'''
import lightgbm as lgb

#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

import time

#Load dataset
cancer = datasets.load_breast_cancer()

start = time.time()

# print the names of the 13 features
print("Features: ", cancer.feature_names)

# print the label type of cancer('malignant', 'benign')
print("Labels: ", cancer.target_names)

Features=  ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness', 'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry', 'mean fractal dimension', 'radius error', 'texture error', 'perimeter error', 'area error', 'smoothness error', 'compactness error', 'concavity error', 'concave points error', 'symmetry error', 'fractal dimension error', 'worst radius', 'worst texture', 'worst perimeter', 'worst area', 'worst smoothness', 'worst compactness', 'worst concavity', 'worst concave points', 'worst symmetry', 'worst fractal dimension']
Labels =  ['malignant','benign']

# print data(feature)shape
print(cancer.data.shape)

# print the cancer labels (0:malignant, 1:benign)
print(cancer.target)

# Split dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target, test_size=0.3,random_state=109) # 70% training and 30% test
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.3,random_state=109) # 70% training and 30% test

lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

params = {
    'objective': 'binary',
    'metric': 'auc',
    'num_iterations': 50,
    'early_stopping_rounds': 50,
    'learning_rate': 0.1,
    # max_depth: 20 # 10, 40
    # num_leaves: 31 # 50, 60
    'boosting': 'gbdt',
    # min_data_in_leaf: 20 # 20, 40
}

bst = lgb.train(params, train_set=lgb_train, valid_sets=[lgb_eval], callbacks=[lgb.log_evaluation()])

#Predict the response for test dataset
y_pred = bst.predict(X_test).round(0)

# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))

print("Time taken :", time.time() - start)