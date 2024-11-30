from import_data_and_preprocessing import X, Y
from sklearn.ensemble import BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pickle

print('\nGait Classification using Bagging with GridSearchCV:\n')

# Train-test split
x, x_t, y, y_t = train_test_split(X, Y, test_size=0.2, random_state=1)

# Define base estimator: SVM or KNN
use_svm = True  # Set to False to use KNN instead of SVM

if use_svm:
    base_clf = svm.SVC(probability=True, random_state=0)  # Use probability=True for Bagging
    param_grid = {
        'estimator__C': [6,8,10],
        'estimator__gamma': ['scale',80,100,120],
        'estimator__kernel': ['rbf'],
        'estimator__decision_function_shape' : ['ovo'],
        'n_estimators': [50, 60, 70, 80, 90],  # Number of bagging estimators
        'max_samples': [0.6, 0.8, 1.0],  # Fraction of training data per estimator
        'max_features': [0.8, 1.0]  # Fraction of features per estimator
    }
    print("Using SVM as the base estimator for Bagging...\n")
else:
    base_clf = KNeighborsClassifier()
    param_grid = {
        'estimator__n_neighbors': [1,2,3,4,5,6],
        'estimator__metric': ['manhattan'],
        'estimator__weights': ['distance'],
        'n_estimators': [50, 60, 70, 80, 90],  # Number of bagging estimators
        'max_samples': [0.6, 0.8, 1.0],  # Fraction of training data per estimator
        'max_features': [0.8, 1.0]  # Fraction of features per estimator
    }
    print("Using KNN as the base estimator for Bagging...\n")

# Initialize BaggingClassifier
bagging_clf = BaggingClassifier(
    estimator=base_clf,
    bootstrap=True,
    bootstrap_features=False,
    random_state=0,
    n_jobs=-1  # Use all available processors
)

# Perform grid search
print('Performing grid search for Bagging Classifier...')
grid_search = GridSearchCV(
    estimator=bagging_clf,
    param_grid=param_grid,
    scoring='accuracy',
    cv=10,  # 10-fold cross-validation
    verbose=3,
    n_jobs=-1
)
grid_search.fit(x, y)

# Best parameters
best_parameters = grid_search.best_params_
print('\nBest parameters for Bagging Classifier:\n', best_parameters)

# Best model from grid search
best_bagging_clf = grid_search.best_estimator_

# Evaluate on the test set
print('\nEvaluating the best Bagging Classifier...')
pred = best_bagging_clf.predict(x_t)
res = best_bagging_clf.score(x_t, y_t)

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
ax.set_title('Confusion Matrix for Best Bagging Classifier', fontsize=15)
ax.set_xlabel('Target labels', fontsize=14)
ax.set_ylabel('Predicted labels', fontsize=14)

plt.show()

# Save the best model
filename = 'best_bagging_classifier.sav'
pickle.dump(best_bagging_clf, open(filename, 'wb'))
