# THE ALGORITHM
import pandas as pd
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

#lol = pd.read_csv('training-data/encoded_data.csv')
train = pd.read_csv('training-data/alldata_notempo.csv')
X = train.drop(columns=['genre'])
y = train['genre']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=309)

#clf = DecisionTreeClassifier()
clf = RandomForestClassifier()
#clf = KNeighborsClassifier()
#clf = svm.SVC(kernel= 'linear')
#clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0)
#clf = MLPClassifier()
# k_fold = KFold(n_splits=3, shuffle=True, random_state=42)
# cross_val_scores = cross_val_score(clf, X_train, y_train, cv=k_fold, scoring='accuracy')

clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
test_data = pd.read_csv('training-data/testing_ready.csv')
X_test = test_data.drop(columns=['instance_id'])
y_pred = clf.predict(X_test)
result_df = pd.DataFrame({
    'instance_id': test_data['instance_id'],
    'genre': y_pred
})
result_df.to_csv('test-data/predicted_genres_newversion.csv', index=False)
