from import_data_and_preprocessing import X, Y
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import svm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

print('\nGait Classification using Stacking Classifier (RF as Base Model and SVM as Meta-Classifier):\n')

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
svm_grid = {
    'gamma': ['scale',0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
    'C': [0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
    'kernel': ['linear', 'rbf', 'poly'],
    'decision_function_shape' : ['ovo', 'ovr']
}

# Initialize basic classifier
rf_clf = RandomForestClassifier(random_state=0)
knn_clf = KNeighborsClassifier()
svm_clf = svm.SVC(probability=True, random_state=0)

rf_search = GridSearchCV(estimator=rf_clf, param_grid=rf_grid, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)
knn_search = GridSearchCV(estimator=knn_clf, param_grid=knn_grid, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)
svm_search = GridSearchCV(estimator=svm_clf, param_grid=svm_grid, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)

print('Performing grid search for Random Forest...')
rf_search.fit(x, y)

# print('Performing grid search for KNN...')
# knn_search.fit(x, y)

# print('Performing grid search for SVM...')
# svm_search.fit(x, y)

# Retrieve the best model
best_rf = rf_search.best_estimator_
print('\nBest Random Forest parameters:', rf_search.best_params_)

# best_knn = knn_search.best_estimator_
# print('Best KNN parameters:', knn_search.best_params_)

# best_svm = svm_search.best_estimator_
# print('Best SVM parameters:', svm_search.best_params_)

# Perform grid search for SVM as the meta-classifier
print('Performing grid search for the Meta-Classifier...')

# Prepare the Stacking Classifier for training
stacking_clf = StackingClassifier(
    estimators=[('rf', best_rf)],  # Use the best RF model as the base model
    # estimators=[('knn', best_knn)],
    # estimators=[('svm', best_svm)],
    final_estimator=svm_search,  # GridSearchCV for the SVM meta-classifier
    cv=5
)

# Train the stacking classifier
print('\nTraining the Stacking Classifier...')
stacking_clf.fit(x, y)

# Evaluate on the test set
print('\nEvaluating the Stacking Classifier...')
pred = stacking_clf.predict(x_t)
res = stacking_clf.score(x_t, y_t)

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
ax.set_title('Confusion Matrix for Stacking Classifier', fontsize=15)
ax.set_xlabel('Target labels', fontsize=14)
ax.set_ylabel('Predicted labels', fontsize=14)

plt.show()

# Save the stacking classifier
filename = 'stacking_classifier_with_svm_meta.sav'
pickle.dump(stacking_clf, open(filename, 'wb'))
