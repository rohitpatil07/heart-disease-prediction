#heart disease prediction using random foest with gridsearchcv for boosting accuracy

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('heart.csv')

#check for missing values
data.isnull().sum()

#Get Target data 
y = data['target']

#Load X Variables into a Pandas Dataframe with columns 
X = data.drop(['target'], axis = 1)

print(f'X : {X.shape}')

#Divide dataset

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=101)

#Build model normal way

rf_Model = RandomForestClassifier()

rf_Model.fit(X_train,y_train)

print (f'Train Accuracy - : {rf_Model.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Model.score(X_test,y_test):.3f}')

#Maybe data is over fitted

#Build model using grid search csv

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 80, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [2,6]
# Minimum number of samples required to split a node
min_samples_split = [2, 7]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]

# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

# rf_Model = RandomForestClassifier('bootstrap'= False,'max_depth'= 8,'max_features'= 'sqrt','min_samples_leaf'= 1,'min_samples_split'= 2,'n_estimators'= 64)
rf_Model = RandomForestClassifier()

from sklearn.model_selection import GridSearchCV
rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 3, verbose=2, n_jobs = 4)

rf_Grid.fit(X_train, y_train)

rf_Grid.best_params_

print (f'Train Accuracy - : {rf_Grid.score(X_train,y_train):.3f}')
print (f'Test Accuracy - : {rf_Grid.score(X_test,y_test):.3f}')