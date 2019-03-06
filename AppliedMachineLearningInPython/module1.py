import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd
from sklearn.model_selection import train_test_split

# read in data
fruits = pd.read_table('data/fruit_data_with_colors.txt')

# create lookup table
lookup_fruit_name = dict(zip(fruits.fruit_label.unique(), fruits.fruit_name.unique()))

# plot scatter plots
"""
X = fruits[['height', 'width', 'mass', 'color_score']]
y = fruits['fruit_label']
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(X_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
plt.show()
"""

# train classifier
X = fruits[['mass', 'width', 'height']]
y = fruits['fruit_label']

# default is 75% / 25% train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# create classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, y_train)

# estimate accuracy 
accuracy = knn.score(X_test, y_test)
#print accuracy

# classify new objects
#fruit_prediction = knn.predict([[20, 4.3, 5.5]])
#print lookup_fruit_name[fruit_prediction[0]]

#fruit_prediction2 = knn.predict([[100, 6.3, 8.5]])
#print lookup_fruit_name[fruit_prediction2[0]]

# plot the decision boundaries of the kNN classifier
from imports.adspy_shared_utilities import plot_fruit_knn
#plot_fruit_knn(X_train, y_train, 5, "uniform")

# check sensitivity to k parameter
k_range = range(1, 20)
scores = []

# loop over k values
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors = k)
    knn.fit(X_train, y_train)
    scores.append(knn.score(X_test, y_test))

# plot accuracy vs k value
plt.figure()
plt.xlabel('k')
plt.ylabel('accuracy')
plt.scatter(k_range, scores)
plt.xticks([0, 5, 10, 15, 20]);
#plt.show()

# check sensitivity to train/test split
t = [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
knn1 = KNeighborsClassifier(n_neighbors = 5)

plt.figure()

for s in t:

    scores = []
    for i in range(1,1000):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1-s)
        knn1.fit(X_train, y_train)
        scores.append(knn1.score(X_test, y_test))
    plt.plot(s, np.mean(scores), 'bo')

plt.xlabel('Training set proportion (%)')
plt.ylabel('accuracy');
plt.show()
