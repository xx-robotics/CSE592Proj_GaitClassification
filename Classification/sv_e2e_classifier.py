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

print('\nGait Classification using Direct Voting Classifier (RF, KNN, and SVM):\n')

# Train-test split
x, x_t, y, y_t = train_test_split(X, Y, test_size=0.2, random_state=1)

# Define hyperparameter grids
param_grid = {
    'rf__n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'rf__max_depth': [5,6,7,8,9,10,11,12,13,14,15],
    'rf__criterion': ['gini', 'entropy'],
    'rf__bootstrap': [True, False],
    'knn__n_neighbors': [2,3,4,5,6,7,8,9,10],
    'knn__metric': ['euclidean', 'manhattan', 'minkowski','chebyshev'],
    'knn__weights': ['uniform', 'distance'],
    'svm__C': [0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
    'svm__gamma': ['scale',0.001,0.01,0.1,1,2,3,4,5,6,7,8,9,10,20,30,40,50,100,500,1000],
    'svm__kernel': ['linear', 'rbf', 'poly'],
    'svm__decision_function_shape' : ['ovo', 'ovr']
}

# Initialize classifiers
rf_clf = RandomForestClassifier(random_state=0)
knn_clf = KNeighborsClassifier()
svm_clf = svm.SVC(probability=True, random_state=0)  # SVM with `probability=True` for soft voting

# Define the Voting Classifier
voting_clf = VotingClassifier(estimators=[
    ('rf', rf_clf),
    ('knn', knn_clf),
    ('svm', svm_clf)
], voting='soft')

# Grid search directly on the Voting Classifier
print('Performing grid search on Voting Classifier...')
grid_search = GridSearchCV(estimator=voting_clf, param_grid=param_grid, scoring='accuracy', cv=10, verbose=3, n_jobs=-1)
grid_search.fit(x, y)

# Best parameters
best_parameters = grid_search.best_params_
print('\nBest parameters for Voting Classifier:', best_parameters)

# Evaluate on the test set
print('\nEvaluating the Voting Classifier...')
pred = grid_search.predict(x_t)
res = grid_search.score(x_t, y_t)

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
filename = 'voting_classifier_e2e.sav'
pickle.dump(grid_search, open(filename, 'wb'))
