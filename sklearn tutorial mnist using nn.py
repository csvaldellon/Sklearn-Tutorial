from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
import mnist, pandas as pd
from tensorflow.keras import layers, Sequential, callbacks, optimizers

x_train = mnist.train_images()
y_train = mnist.train_labels()
x_test = mnist.test_images()
y_test = mnist.test_labels()
print(x_train)
x_train = x_train.reshape((-1, 28*28))
print(x_train)
x_test = x_test.reshape((-1, 28*28))
x_train = x_train/256
x_test = x_test/256

clf = MLPClassifier(solver='adam', activation='relu', hidden_layer_sizes=(64, 64))
clf.fit(x_train, y_train)
prediction = clf.predict(x_test)
accuracy_1 = confusion_matrix(y_test, prediction).trace()/confusion_matrix(y_test, prediction).sum()

dnn = Sequential([
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
dnn.compile(loss='sparse_categorical_crossentropy', optimizer=optimizers.Adam(0.0001))
dnn.fit(x_train, y_train, epochs=100, validation_split=0.2, verbose=0,
        callbacks=[callbacks.EarlyStopping(patience=10, monitor='val_loss')])
predictions = dnn.predict(x_test)
predictions = np.array([np.argmax(predictions[i]) for i in range(len(y_test))])
count = 0
for i in range(len(predictions)):
    if predictions[i] - np.array(y_test)[i] == 0:
        count += 1

print("accuracy using sklearn dnn: ", accuracy_1)
print("accuracy using keras dnn: ", count/len(predictions))
