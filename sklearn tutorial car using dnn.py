import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt

data = pd.read_csv('car')

data['buying'] = (data['buying'] == 'low')*1.0 + \
                 (data['buying'] == 'med')*2.0 + \
                 (data['buying'] == 'high')*3.0 + \
                 (data['buying'] == 'vhigh')*4.0

data.pop('doors')     # removed for now
data.pop('persons')   # removed for now

data['maint'] = (data['maint'] == 'low')*1.0 + \
                (data['maint'] == 'med')*2.0 + \
                (data['maint'] == 'high')*3.0 + \
                (data['maint'] == 'vhigh')*4.0
data['lug_boot'] = (data['lug_boot'] == 'small')*1.0 + \
                   (data['lug_boot'] == 'med')*2.0 + \
                   (data['lug_boot'] == 'big')*3.0
# data.pop('lug_boot')
data['safety'] = (data['safety'] == 'low')*1.0 + \
                 (data['safety'] == 'med')*2.0 + \
                 (data['safety'] == 'high')*3.0

data['class'] = (data['class'] == 'unacc')*1 + \
                 (data['class'] == 'acc')*2 + \
                 (data['class'] == 'good')*3 + \
                 (data['class'] == 'vgood')*4

# data.to_csv('C:/Users/Val/Desktop/car_processed_2.csv')

y = data.pop('class')
x = data

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
# print(x_train)
# print(y_train)

x_train = tf.keras.utils.normalize(x_train, axis=1)
# print(x_train)
x_test = tf.keras.utils.normalize(x_test, axis=1)
# print(x_test)


def build_model():
    model = keras.Sequential([
        layers.Dense(512, activation=tf.nn.relu, input_shape=[len(x_train.keys())]),
        layers.Dense(256, activation=tf.nn.relu),
        layers.Dense(10, activation=tf.nn.softmax)
    ])

    LR = 0.0001
    optimizer = tf.keras.optimizers.Adam(LR)

    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


model = build_model()


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


early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

history = model.fit(x_train, y_train, epochs=EPOCHS, validation_split=0.2,
                    callbacks=[early_stop])
print("")
plot_history(history)
plt.show()
model.save('sklearn class knn.model')

# model = keras.models.load_model('sklearn class knn.model')
test_predictions = model.predict([x_test])
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
