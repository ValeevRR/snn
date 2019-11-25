import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_all = np.concatenate((x_test, x_train), axis=0)
x_train = x_all[:51100]
print("Размер x_train = ", len(x_train))
x_test = x_all[51100:]
print("Размер x_test = ", len(x_test))
y_all = np.concatenate((y_test, y_train), axis=0)
y_train = y_all[:51100]
print("Размер y_train = ", len(y_train))
y_test = y_all[51100:]
print("Размер y_test = ", len(y_test))
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
x_train.shape
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(' x_train shape: ', x_train.shape)
print(' Количество изображений в x_train ', x_train.shape[0])
print(' Количество изображений в x_test ', x_test.shape[0])
model = Sequential()
model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(10, activation=tf.nn.softmax))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=x_train, y=y_train, epochs=10)
model.evaluate(x_test, y_test)
image_index = 4444
plt.imshow(x_test[image_index].reshape(28, 28), cmap='Greys')
pred = model.predict(x_test[image_index].reshape(1, 28, 28, 1))
model.save('saved.h5')
print(pred.argmax())
