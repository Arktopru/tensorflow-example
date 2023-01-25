import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from keras import layers

x_train = tf.cast([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], tf.float32)
y_train = tf.cast([0, 1, 1, 0], tf.float32)
x_validation = tf.cast([
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0]
], tf.float32)
y_validation = tf.cast([0, 0, 1, 1], tf.float32)
x_test = tf.cast([
    [0, 0],
    [0, 1],
    [1, 1],
    [1, 0]
], tf.float32)
y_test = tf.cast([0, 1, 0, 1], tf.float32)

model = keras.Sequential([
    keras.layers.Dense(16, input_dim=2, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.summary()

model.compile(
    loss='mean_squared_error',
    optimizer='adam',
    metrics=["binary_accuracy"]
)

history = model.fit(
    x=x_train, y=y_train,
    epochs=500,
    verbose=0
)
loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy: ", accuracy)

predictions = model.predict(x_test).round()
print(predictions)

# keras.utils.plot_model(model, "my_first_model.png")
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
