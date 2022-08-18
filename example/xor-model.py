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
x_test = tf.cast([
    [0, 0],
    [1, 1],
    [0, 1],
    [1, 0]
], tf.float32)
y_test = tf.cast([0, 0, 1, 1], tf.float32)

model = keras.Sequential([
    keras.layers.Input(shape=(2,)),
    keras.layers.Dense(units=2, activation='relu'),
    keras.layers.Dense(units=1, activation='sigmoid')
])
model.summary()

model.compile(
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    optimizer='adam',
    metrics=["accuracy"]
)

history = model.fit(
    x=x_train, y=y_train,
    epochs=1000,
    steps_per_epoch=1,
    validation_data=(x_test, y_test),
    validation_steps=1
)

predictions = model.predict(x_test)

print(predictions)

# keras.utils.plot_model(model, "my_first_model.png")
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
