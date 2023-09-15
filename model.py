# THE ALGORITHM
import pandas as pd
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#lol = pd.read_csv('training-data/encoded_data.csv')
lol = pd.read_csv('training-data/encoded_data.csv')

# # Split the data into training and testing sets
X = lol.drop(columns=['genre'])
y = lol['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.61862, random_state=309)

# # Create and train a Decision Tree Classifier
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = KNeighborsClassifier()
#clf = svm.SVC(kernel= 'linear')
<<<<<<< Updated upstream
clf = gradient_boosting_classifier = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)
=======
clf = GradientBoostingClassifier()
#clf = MLPClassifier()
# k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
# cross_val_scores = cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')

>>>>>>> Stashed changes
clf.fit(X_train, y_train)

# # Make predictions on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
#print(classification_report(y_test, y_pred))

test_data = pd.read_csv('test_mod.csv')

X_test = test_data.drop(columns=['instance_id'])
y_pred = clf.predict(X_test)

# Create a DataFrame with instance IDs and predicted genres
result_df = pd.DataFrame({
    'instance_id': test_data['instance_id'],
    'genre': y_pred
})
<<<<<<< Updated upstream

# Save the result DataFrame to a CSV file
result_df.to_csv('test-data/predicted_genres.csv', index=False)



# import pandas as pd
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Load the training data
# train_data = pd.read_csv('training-data/encoded_data.csv')

# # Split the training data into features (X) and target (y)
# X_train = train_data.drop(columns=['genre'])
# y_train = train_data['genre']

# # Load the test data
# test_data = pd.read_csv('test_mod.csv')

# # Split the test data into features (X_test) and instance IDs
# X_test = test_data.drop(columns=['instance_id'])

# # Create and train a Random Forest Classifier
# clf = RandomForestClassifier(n_estimators=100, random_state=309)
# clf.fit(X_train, y_train)

# # Make predictions on the test set
# y_pred = clf.predict(X_test)

# # Create a DataFrame with instance IDs and predicted genres
# result_df = pd.DataFrame({
#     'instance_id': test_data['instance_id'],
#     'predicted_genre': y_pred
# })

# # Save the result DataFrame to a CSV file
# result_df.to_csv('test-data/predicted_genres.csv', index=False)
=======
result_df.to_csv('test-data/predicted_genres_newversion.csv', index=False)


# import numpy as np
# import pandas as pd
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score
# from sklearn.model_selection import KFold, train_test_split
# from sklearn.metrics import classification_report

# # Load the data and split into features and target
# train = pd.read_csv('training-data/alldata_notempo.csv')
# X = train.drop(columns=['genre'])
# y = train['genre']

# # Split the data into training and testing sets (for final evaluation)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=309)

# # Initialize the classifier
# clf = GradientBoostingClassifier()

# # Define the number of folds for K-fold cross-validation
# n_splits = 5  # You can adjust this number as needed

# # Initialize K-fold cross-validator
# k_fold = KFold(n_splits=n_splits, shuffle=True, random_state=42)

# # Lists to store cross-validation results
# cross_val_scores = []

# # Perform K-fold cross-validation
# for train_indices, val_indices in k_fold.split(X_train):
#     X_train_fold, X_val_fold = X_train.iloc[train_indices], X_train.iloc[val_indices]
#     y_train_fold, y_val_fold = y_train.iloc[train_indices], y_train.iloc[val_indices]
    
#     clf.fit(X_train_fold, y_train_fold)
#     y_pred_fold = clf.predict(X_val_fold)
#     accuracy_fold = accuracy_score(y_val_fold, y_pred_fold)
#     cross_val_scores.append(accuracy_fold)

# # Calculate the mean and standard deviation of cross-validation scores
# mean_cv_score = sum(cross_val_scores) / len(cross_val_scores)
# std_cv_score = np.std(cross_val_scores)

# # Print the cross-validation results
# print(f'Mean CV Accuracy: {mean_cv_score:.2f}')
# print(f'Standard Deviation of CV Accuracy: {std_cv_score:.2f}')

# # Train the final model on the entire training set and evaluate on the test set
# clf.fit(X_train, y_train)
# y_pred_test = clf.predict(X_test)
# test_accuracy = accuracy_score(y_test, y_pred_test)
# print(f'Test Accuracy: {test_accuracy:.2f}')

# # Save the final model's predictions on the test set
# test_data = pd.read_csv('training-data/testing_ready.csv')
# X_test_final = test_data.drop(columns=['instance_id'])
# y_pred_final = clf.predict(X_test_final)
# result_df = pd.DataFrame({
#     'instance_id': test_data['instance_id'],
#     'genre': y_pred_final
# })
# result_df.to_csv('test-data/predicted_genres_newversion.csv', index=False)

>>>>>>> Stashed changes
