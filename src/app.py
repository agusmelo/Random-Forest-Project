import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score


df = pd.read_csv('/workspace/Random-Forest-Project/data/processed/df_processed.csv')

X = df.drop(columns = ['Survived']).copy()
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train,y_train)


# GridSearch

clf2 = RandomForestClassifier(random_state= 42)

max_depth = [6, 8, 10, 12, 14, 16, 18]
min_samples_split = [8, 12, 16, 20, 24]
criterion = ['gini', 'entropy']

grid = dict(max_depth = max_depth, min_samples_split = min_samples_split, criterion = criterion)
grid_search = GridSearchCV(estimator = clf2, param_grid = grid, n_jobs = -1, cv = 5)
grid_search_result = grid_search.fit(X, y)

clf_gscv = grid_search_result.best_estimator_


# RandomizedSearch

clf3 = RandomForestClassifier(random_state= 42)

n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]
criterion = ['gini','entropy']
class_weight = ['balanced', None]

grid_r = dict(n_estimators = n_estimators, max_features = max_features, max_depth = max_depth, min_samples_split = min_samples_split, min_samples_leaf = min_samples_leaf, bootstrap = bootstrap, criterion = criterion, class_weight = class_weight)
grid_random=RandomizedSearchCV(estimator = clf3, n_iter = 100, cv = 5, random_state = 42, param_distributions = grid_r)
grid_random_result = grid_random.fit(X_train, y_train)

clf_rscv = grid_random_result.best_estimator_



filename1 = '/workspace/Random-Forest-Project/models/model_rf.sav'
pickle.dump(clf, open(filename1, 'wb'))

filename2 = '/workspace/Random-Forest-Project/models/model_rf_gscv.sav'
pickle.dump(clf2, open(filename2, 'wb'))

filename3 = '/workspace/Random-Forest-Project/models/model_rf_rscv.sav'
pickle.dump(clf3, open(filename3, 'wb'))