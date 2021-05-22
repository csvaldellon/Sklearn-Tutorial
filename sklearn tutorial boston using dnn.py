from sklearn import datasets, linear_model, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt, tensorflow as tf, pandas as pd, numpy as np
from tensorflow import keras
from tensorflow.keras import layers, optimizers, utils, callbacks, models

boston = datasets.load_boston()
x = boston.data
y = boston.target
x_df = pd.DataFrame(x)
y_df = pd.DataFrame(y)

x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2)
x_train = utils.normalize(x_train, axis=1)
x_test = utils.normalize(x_test, axis=1)

lr = 0.01
EPOCHS = 1000
early_stop = callbacks.EarlyStopping(monitor='val_loss', patience=10)
# model = keras.Sequential([
    # layers.Dense(64, activation=tf.nn.relu, input_shape=[len(x_train.keys())]),
    # layers.Dense(64, activation=tf.nn.relu),
    # layers.Dense(1)
# ])
# model.compile(loss='mse', optimizer=optimizers.Adam(lr), metrics=['mse', 'mae'])
# model.fit(x_train, y_train, epochs=EPOCHS, callbacks=[early_stop], validation_split=0.2)
# model.save("boston dnn.model")
model = models.load_model("boston dnn.model")

prediction = [model.predict(x_test)[i][0] for i in range(len(y_test))]
y_test = [np.array(y_test)[i][0] for i in range(len(y_test))]

plt.scatter(y_test, prediction)
plt.xlabel("actual")
plt.ylabel("prediction")
plt.plot([0, 60], [0, 60])
plt.show()

r_squared = metrics.r2_score(y_test, prediction)
print(r_squared)
