from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt, pandas as pd, numpy as np

boston = datasets.load_boston()
x = boston.data
y = boston.target

l_reg = linear_model.LinearRegression()

# plt.scatter(x.T[5], y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
model = l_reg.fit(x_train, y_train)
predictions = model.predict(x_test)
print(x_test)
# print("Predictions: ", predictions)
# print("R^2 value: ", l_reg.score(x, y))
r_squared = metrics.r2_score(y_test, predictions)
print(r_squared)

plt.scatter(y_test, predictions)
plt.xlabel("actual")
plt.ylabel("prediction")
plt.plot([0, 60], [0, 60])
plt.show()
