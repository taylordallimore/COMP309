# THE ALGORITHM
import pandas as pd
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


#lol = pd.read_csv('training-data/encoded_data.csv')
lol = pd.read_csv('training-data/encoded_data.csv')

# # Split the data into training and testing sets
X = lol.drop(columns=['genre'])
y = lol['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=309)
#test_size=0.61862

# # Create and train a Decision Tree Classifier
#clf = DecisionTreeClassifier()
#clf = RandomForestClassifier()
#clf = KNeighborsClassifier()
#clf = svm.SVC(kernel= 'linear')
clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=2, random_state=309)
k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
cross_val_scores = cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')

clf.fit(X_train, y_train)

# # Make predictions on the test set
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

# accuracy = accuracy_score(y_test, y_pred)
# print(f'Accuracy on Test Set: {accuracy:.2f}')
# Print cross-validation results
# print(f'Cross-Validation Scores: {cross_val_scores}')
# print(f'Mean Cross-Validation Score: {cross_val_scores.mean():.2f}')
#print(classification_report(y_test, y_pred))

test_data = pd.read_csv('test_mod.csv')

X_test = test_data.drop(columns=['instance_id'])
y_pred = clf.predict(X_test)

# Create a DataFrame with instance IDs and predicted genres
result_df = pd.DataFrame({
    'instance_id': test_data['instance_id'],
    'genre': y_pred
})

# Save the result DataFrame to a CSV file
result_df.to_csv('test-data/predicted_genres.csv', index=False)
