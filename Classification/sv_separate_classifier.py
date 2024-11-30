from import_data_and_preprocessing import X, Y
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

print('\nGait Classification using Voting Classifier (RF, KNN, and SVM):\n')

# Train-test split
x, x_t, y, y_t = train_test_split(X, Y, test_size=0.2, random_state=1)

# Define hyperparameter grids
rf_grid = {
    'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'max_depth': [5,6,7,8,9,10,11,12,13,14,15],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}
knn_grid = {
    'n_neighbors': [2,3,4,5,6,7,8,9,10],
    'metric': ['euclidean', 'manhattan', 'minkowski','chebyshev'],
    'weights': ['uniform', 'distance']
}
# svm_grid = {
#     'gamma': ['scale',0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
#     'C': [0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
#     'kernel': ['linear', 'rbf', 'poly'],
#     'decision_function_shape' : ['ovo', 'ovr']
# }

# Initialize classifiers
rf_clf = RandomForestClassifier(random_state=0)
knn_clf = KNeighborsClassifier()
# svm_clf = svm.SVC(probability=True, random_state=0)  # SVM with `probability=True` for soft voting

# Perform grid search for each classifier
print('Performing grid search for Random Forest...')
rf_search = GridSearchCV(estimator=rf_clf, param_grid=rf_grid, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)
rf_search.fit(x, y)

print('Performing grid search for KNN...')
knn_search = GridSearchCV(estimator=knn_clf, param_grid=knn_grid, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)
knn_search.fit(x, y)

# print('Performing grid search for SVM...')
# svm_search = GridSearchCV(estimator=svm_clf, param_grid=svm_grid, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)
# svm_search.fit(x, y)

# Retrieve best estimators
best_rf = rf_search.best_estimator_
best_knn = knn_search.best_estimator_
# best_svm = svm_search.best_estimator_

# Print best parameters and CV scores
print('\nBest Random Forest parameters:', rf_search.best_params_)
print('Best KNN parameters:', knn_search.best_params_)
# print('Best SVM parameters:', svm_search.best_params_)

# Define voting classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', best_rf),
    ('knn', best_knn),
    # ('svm', best_svm)
], voting='soft', weights=[1, 0.4])

# Fit voting classifier on the training data
print('\nTraining the Voting Classifier...')
voting_clf.fit(x, y)

# Evaluate on the test set
print('\nEvaluating the Voting Classifier...')
pred = voting_clf.predict(x_t)
res = voting_clf.score(x_t, y_t)

# Results
print('\nClassification accuracy on test set: ', round(res, 4))
conf_matrix = confusion_matrix(y_t, pred)
print('Confusion Matrix:\n', conf_matrix)
print('\nClassification Report:\n', classification_report(y_t, pred))

# Heatmap of confusion matrix
df_cm = pd.DataFrame(conf_matrix, index=['CA', 'HSP', 'PD', 'HC'],
                     columns=['CA', 'HSP', 'PD', 'HC'])
ax = plt.axes()
sns.heatmap(df_cm, cmap='BuPu', linewidths=2, square=False, annot=True)
ax.set_title('Confusion Matrix for Voting Classifier', fontsize=15)
ax.set_xlabel('Target labels', fontsize=14)
ax.set_ylabel('Predicted labels', fontsize=14)

plt.show()

# Save the voting classifier
filename = 'voting_classifier_separate.sav'
pickle.dump(voting_clf, open(filename, 'wb'))
