import keras
import tensorflow as tf
import pandas as pd
import numpy as np

from keras import layers

# column_names = ['i1', 'i2', 'o1']
#
# raw_dataset = pd.read_csv('data/data.csv', names=column_names,
#                           na_values='?', sep=',', skipinitialspace=True)
#
# dataset = raw_dataset.copy()
# dataset.tail()
train_data = tf.constant([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
print(train_data)
validation_data = tf.constant([
    1, 0, 0, 1
])
inputs = keras.Input(shape=(4, 2), dtype='int32', name='input_layer_1')

dense = layers.Dense(units=1, name="output_layer_1")
outputs = dense(inputs)

model = keras.Model(inputs, outputs)
model.summary()

model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"]
)

history = model.fit(train_data, validation_data, batch_size=4, epochs=10)
test_data = train_data

model.predict(test_data, verbose=2)
# keras.utils.plot_model(model, "my_first_model.png")
# keras.utils.plot_model(model, "my_first_model_with_shape_info.png", show_shapes=True)
