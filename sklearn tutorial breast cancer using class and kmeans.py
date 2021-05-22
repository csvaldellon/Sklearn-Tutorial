from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
from tensorflow.keras import Sequential, layers, optimizers, utils, callbacks
from sklearn import neighbors, svm, cluster

bc = load_breast_cancer()
x = bc.data
y = bc.target
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn = neighbors.KNeighborsClassifier(n_neighbors=23, weights='uniform')
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy_1 = accuracy_score(y_test, prediction)

SVM = svm.SVC()
SVM.fit(x_train, y_train)
prediction = SVM.predict(x_test)
accuracy_2 = accuracy_score(y_test, prediction)

x_train = utils.normalize(x_train)
x_test = utils.normalize(x_test)
dnn = Sequential([
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])
lr = 0.001
dnn.compile(loss='binary_crossentropy', optimizer=optimizers.Adam(lr), metrics=['accuracy'])
EPOCHS = 1000
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
dnn.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2, callbacks=[early_stop], verbose=0)
predictions = [round(dnn.predict(x_test).T[0][i]) for i in range(len(y_test))]
count = 0
for i in range(len(predictions)):
    if predictions[i] - np.array(y_test)[i] == 0:
        count += 1

print("accuracy using knn: ", accuracy_1)
print("accuracy using svm: ", accuracy_2)
print("accuracy using dnn: ", count/len(predictions))

k_means = cluster.KMeans(n_clusters=2, random_state=123)
k_means.fit(x_train)
predictions = k_means.predict(x_test)
accuracy = accuracy_score(y_test, predictions)
accuracy = max(1-accuracy, accuracy)
print("accuracy using kmeans: ", accuracy)
