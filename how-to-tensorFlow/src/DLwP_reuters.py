import imp
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt

reuters = keras.datasets.reuters

(datas_train, labels_train), (datas_test,
                              labels_test) = reuters.load_data(num_words=10000)


# %%
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


# %%
x_train = vectorize_sequences(datas_train)
x_test = vectorize_sequences(datas_test)

y_train = to_categorical(labels_train)
y_test = to_categorical(labels_test)

# %%
model = keras.models.Sequential([
    layers.Dense(256, activation='relu', input_shape=(10000, )),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(46, activation='softmax')
])

model.compile(optimizer='Adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# %%
# model.summary()
history = model.fit(x_train,
                    y_train,
                    epochs=6,
                    validation_data=(x_test, y_test))

# %%在新数据上生成预测结果
predictions = model.predict(x_test)
np.argmax(predictions[2])
