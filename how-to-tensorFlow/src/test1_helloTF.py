import tensorflow as tf
# import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# print("GPU", tf.test.is_gpu_available())

# a = tf.constant(2.)
# b = tf.constant(4.)

# print(a * b)

mnist = tf.keras.datasets.mnist

(datas_train, labels_train), (datas_test, labels_test) = mnist.load_data()
x_train = tf.convert_to_tensor(datas_train, dtype=tf.float32) / 255
x_test = tf.convert_to_tensor(datas_test, dtype=tf.float32) / 255
# x_train, x_test = datas_train / 255.0, datas_test / 255.0

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='SGD',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train,
          labels_train,
          epochs=8,
          validation_data=(datas_test, labels_test))

# model.evaluate(x_test, labels_test, verbose=2)
