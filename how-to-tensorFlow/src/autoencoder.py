# %%
import matplotlib.pyplot as plt
from enum import auto
from pickle import encode_long
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# %%
x_train = tf.reshape(x_train/255., [-1, 28*28, 1])
x_test = tf.reshape(x_test/255., [-1, 28*28, 1])
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# %%
inputs = layers.Input(shape=(x_train.shape[1],), name='inputs')
x = layers.Dense(32, activation='relu', name='layer_1')(inputs)
outputs = layers.Dense(
    x_train.shape[1], activation='softmax', name='outputs')(x)

auto_encoder = keras.Model(inputs, outputs)
auto_encoder.summary()

# %%
encoder = keras.Model(inputs, x)
encoder.summary()

# %%
decoder_input = keras.Input((32,))
decoder_output = auto_encoder.layers[-1](decoder_input)
decoder = keras.Model(decoder_input, decoder_output)


# %%
auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')
history = auto_encoder.fit(
    x_train, x_train, batch_size=64, epochs=100, validation_split=0.1)

# auto_encoder.save('model/encoder.h5')

# %%
encoded = encoder.predict(x_test)
decoded = decoder.predict(encoded)


# %%
plt.figure(figsize=(10, 4))
# %%
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(np.array(tf.reshape(x_test[5+i], [28, 28])))

    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n+i+1)
    plt.imshow(np.array(tf.reshape(decoded[5+i], [28, 28])))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
