from sklearn import datasets
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()
# plt.imshow(iris['data'])
# plt.show()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)


def build_model():
    model = keras.Sequential([
        layers.Flatten(),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(64, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax)
    ])

    LR = 0.001
    optimizer = tf.keras.optimizers.Adam(LR)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


# model = build_model()


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


EPOCHS = 1000


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(hist['epoch'], hist['loss'], label='Train Error')
    plt.plot(hist['epoch'], hist['val_loss'], label='Val Error')
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(hist['epoch'], hist['accuracy'], label='Train Accuracy')
    plt.plot(hist['epoch'], hist['val_accuracy'], label='Val Accuracy')
    plt.legend()


# early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

# history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2)
print("")
# plot_history(history)
# plt.show()
# model.save('sklearn class.model')
model = keras.models.load_model('sklearn class.model')
test_predictions = model.predict(x_test)
predictions = np.array([np.argmax(test_predictions[i]) for i in range(len(y_test))])
# print(predictions)
# print(y_test)
error = predictions - np.array(y_test)
plt.hist(error, bins=25)
plt.xlabel('Prediction Error')
plt.ylabel('Count')
# plt.show()
count = 0
for i in range(len(predictions)):
    if predictions[i] - np.array(y_test)[i] == 0:
        count += 1

print("accuracy: ", count/len(predictions))
# print(len(predictions))
