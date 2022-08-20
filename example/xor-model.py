import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from keras import layers

x_train = tf.cast([
    [0.22, 0.22],
    [0.22, 1.22],
    [1.22, 0.22],
    [1.22, 1.22],
    [0.1, 0.1],
    [0.1, 1.1],
    [1.1, 0.1],
    [1.1, 1.1],
    [0.75, 0.75],
    [0.75, 1.75],
    [1.75, 0.75],
    [1.75, 1.75],
    [0.78, 0.78],
    [0.78, 1.78],
    [1.78, 0.78],
    [1.78, 1.78],
    [0.7, 0.7],
    [0.7, 1.7],
    [1.7, 0.7],
    [1.7, 1.7]
], tf.float32)
x_train = 1 / x_train
y_train = tf.cast([0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0], tf.float32)
x_validation = tf.cast([
    [0.6, 0.6],
    [1.6, 1.6],
    [0.6, 1.6],
    [1.6, 0.6],
    [0.9, 0.9],
    [1.9, 1.9],
    [0.9, 1.9],
    [1.9, 0.9],
    [0.91, 0.91],
    [1.91, 1.91],
    [0.91, 1.91],
    [1.91, 0.91]
], tf.float32)
x_validation = 1 / x_validation
y_validation = tf.cast([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], tf.float32)
x_test = tf.cast([
    [0.5, 0.5],
    [1.5, 1.5],
    [0.5, 1.5],
    [1.5, 0.5],
    [0.2, 0.2],
    [1.2, 1.2],
    [0.2, 1.2],
    [1.2, 0.2],
    [0.31, 0.31],
    [1.31, 1.31],
    [0.31, 1.31],
    [1.31, 0.31]
], tf.float32)
x_test = 1 / x_test
y_test = tf.cast([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], tf.float32)

model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(units=2, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=["accuracy"]
)

history = model.fit(
    x=x_train, y=y_train,
    epochs=48,
    batch_size=4,
    steps_per_epoch=1,
    validation_data=(x_validation, y_validation),
    validation_steps=1
)
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy)

predictions = model.predict(x_test, batch_size=4)
print(predictions)

# keras.utils.plot_model(model, "my_first_model.png")
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
