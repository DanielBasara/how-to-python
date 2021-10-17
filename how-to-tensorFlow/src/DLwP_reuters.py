import imp
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras import layers
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import datetime

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
model.summary()
history = model.fit(x_train,
                    y_train,
                    epochs=6,
                    validation_data=(x_test, y_test))

# %%在新数据上生成预测结果
predictions = model.predict(x_test)
np.argmax(predictions[2])

# %% 监听数据 tensorboard --logdir project211011
loss = history.history['loss']
train_accuracy = history.history['accuracy']
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = r'C:\Users\11549\OneDrive - Office\data\tensorboard\project211011\logs' + current_time
for i in range(6):
    summary_writer = tf.summary.create_file_writer(log_dir)
    with summary_writer.as_default():
        tf.summary.scalar('loss', float(loss[i]), step=i+1)
        tf.summary.scalar('accuracy', float(train_accuracy[i]), step=i+1)

# %% tensorboard --logdir project211011
mnist = keras.datasets.mnist
(train_datas, train_labels), (test_datas, test_labels) = mnist.load_data()
sample_img = train_datas[0:10]
sample_img = tf.reshape(sample_img, [-1, 28, 28, 1])
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = r'C:\Users\11549\OneDrive - Office\data\tensorboard\project211011\image\img' + current_time 
summary_writer = tf.summary.create_file_writer(log_dir)    
for i in range(10):    
    with summary_writer.as_default():
        tf.summary.image("Training sample", sample_img, max_outputs=10, step=0)
