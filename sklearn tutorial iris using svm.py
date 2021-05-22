from sklearn import datasets, svm, metrics
import numpy as np
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

x = iris.data
y = iris.target

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

model = svm.SVC()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model.fit(x_train, y_train)
predictions = model.predict(x_test)
accuracy = metrics.accuracy_score(y_test, predictions)

print('accuracy', accuracy)
