# %%
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import callbacks
import tensorflow.keras.layers as layers
(datas_train, labels_train), (datas_test,
                              labels_test) = keras.datasets.fashion_mnist.load_data()

# %%
x_train = tf.reshape(datas_train/255., [-1, 28, 28, 1])
x_test = tf.reshape(datas_test/255., [-1, 28, 28, 1])
x_train = tf.cast(x_train, tf.float32)
x_test = tf.cast(x_test, tf.float32)

# %%
inputs = layers.Input(shape=(28, 28, 1), name='input')
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
x = layers.BatchNormalization()(x)
x = layers.MaxPooling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
x = layers.BatchNormalization()(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
outputs = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

CAE = keras.Model(inputs, outputs)
CAE.compile(optimizer='Adam',
            loss='binary_crossentropy',
            )
CAE.summary()
# %%
callbacks_list = [
    keras.callbacks.EarlyStopping(monitor='loss', patience=2),
    keras.callbacks.TensorBoard(
        log_dir='logs',
        histogram_freq=1,
        embeddings_freq=1
    )
]
CAE.fit(x_train, x_train, batch_size=64, epochs=5, validation_split=0.1,
        validation_freq=10, callbacks=callbacks_list)


# %%
results = CAE.predict(x_test[:100])

# %%
plt.figure(figsize=(10, 4))
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(np.array(tf.reshape(x_test[30+i], [28, 28])))

    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n+i+1)
    plt.imshow(np.array(tf.reshape(results[30+i], [28, 28])))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# %% 噪声
noise_factor = 0.4
x_test_noisy = x_test+noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_test.shape)

x_test_noisy = np.clip(x_test_noisy, 0., 1.)

x_train_noisy = x_train+noise_factor * \
    np.random.normal(loc=0.0, scale=1.0, size=x_train.shape)

x_train_noisy = np.clip(x_train_noisy, 0., 1.)
# %%

CAE.fit(x_train_noisy, x_train, batch_size=64, epochs=5, validation_split=0.1, validation_freq=10,
        callbacks=callbacks_list)


# %%
plt.figure(figsize=(10, 4))
n = 5
for i in range(n):
    ax = plt.subplot(2, n, i+1)
    plt.imshow(np.array(tf.reshape(x_test[i], [28, 28])))

    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, n+i+1)
    plt.imshow(np.array(tf.reshape(x_test_noisy[i], [28, 28])))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


# %%
results = CAE.predict(x_test_noisy[:100])


# %%
plt.figure(figsize=(20, 8))
n = 10
for i in range(n):
    ax = plt.subplot(3, n, i+1)
    plt.imshow(np.array(tf.reshape(x_test_noisy[10+i], [28, 28])))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(3, n, n+i+1)
    plt.imshow(np.array(tf.reshape(results[10+i], [28, 28])))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax = plt.subplot(3, n, n+n+i+1)
    plt.imshow(np.array(tf.reshape(x_test[10+i], [28, 28])))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
