# https://www.tensorflow.org/tutorials/keras/text_classification
# Восстановление модели из файлов

import tensorflow as tf
import os
import numpy as np
import shutil
from _defs import custom_standardization

from keras import layers
from keras import losses

print(tf.__version__)

model = tf.keras.models.load_model('saved_model/text_model')
model.load_weights('saved_model/text_model_weights')

url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"

dataset = tf.keras.utils.get_file("aclImdb_v1", url,
                                  untar=True, cache_dir='.',
                                  cache_subdir='')

dataset_dir = os.path.join(os.path.dirname(dataset), 'aclImdb')

train_dir = os.path.join(dataset_dir, 'train')

remove_dir = os.path.join(train_dir, 'unsup')
shutil.rmtree(remove_dir)

batch_size = 32
seed = 42

raw_test_ds = tf.keras.utils.text_dataset_from_directory(
    'aclImdb/test_my',
    batch_size=batch_size)

max_features = 10000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

train_text = raw_test_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


test_ds = raw_test_ds.map(vectorize_text)

AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

predictions = model.predict(test_ds)

for i in [0, 1, 2]:
    print(predictions[i])

# 782/782 [==============================] - 3s 4ms/step - loss: 0.3101 - binary_accuracy: 0.8735
# [[0.65179837]
#  [0.474321  ]
#  [0.38477996]]
