from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import neighbors, metrics


iris = datasets.load_iris()

x = iris.data
y = iris.target

knn = neighbors.KNeighborsClassifier(n_neighbors=13, weights='uniform')
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test, prediction)

print('accuracy', accuracy)
