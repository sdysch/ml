import sklearn
from sklearn import datasets
from sklearn import svm

cancer = datasets.load_breast_cancer()
#print cancer.feature_names
#print cancer.target_names

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(cancer.data, cancer.target, test_size = 0.2)

classes = ["malignant", "benign"]

clf = svm.SVC(kernel = "linear", C = 1)
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)

accuracy = sklearn.metrics.accuracy_score(y_test, y_pred)

print accuracy
