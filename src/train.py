import tensorflow as tf
from tensorflow import keras
import numpy as np
from src.dataload import load_data
import os
import matplotlib.pyplot as plt


image_dir_path = os.getenv("HOME") + "/PycharmProjects/gawibawibo/datasets/train"
(x_train, y_train) = load_data(image_dir_path)
x_train_norm = x_train / 255.0  # 입력은 0~1 사이의 값으로 정규화

# print("x_train shape: {}".format(x_train.shape))
# print("y_train shape: {}".format(y_train.shape))


# plt.imshow(x_train[0])
# print('라벨: ', y_train[101])


model=keras.models.Sequential()
model.add(keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(32, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(32, activation='relu'))
model.add(keras.layers.Dense(3, activation='softmax'))

#
# model=keras.models.Sequential()
# model.add(keras.layers.Conv2D(18, (3,3), activation='relu', input_shape=(28,28,3)))
# model.add(keras.layers.MaxPool2D(2,2))
#
# model.add(keras.layers.Conv2D(36, (3,3), activation='relu'))
# #model.add(keras.layers.MaxPooling2D((3,3)))
#
# model.add(keras.layers.Conv2D(72, (3,3), activation='relu'))
# #model.add(keras.layers.MaxPooling2D((2,2)))
#
#
# model.add(keras.layers.Conv2D(144, (3,3), activation='relu'))
# #model.add(keras.layers.MaxPooling2D((2,2)))
#
# model.add(keras.layers.Conv2D(288, (3,3), activation='relu'))
#
# #################################
#
# # model.add(keras.layers.Conv2D(144, (3,3), activation='relu'))
# # model.add(keras.layers.MaxPooling2D((2,2)))
#
#
#
#
# model.add(keras.layers.Dropout(0.4))
# model.add(keras.layers.Flatten())
# model.add(keras.layers.Dense(26, activation='relu'))
# model.add(keras.layers.Dense(3, activation='softmax'))

model.summary()

model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(x_train, y_train, epochs=100)

# print("Before Reshape - x_train_norm shape: {}".format(x_train_norm.shape))
# # print("Before Reshape - x_test_norm shape: {}".format(x_test_norm.shape))
#
# x_train_reshaped=x_train_norm.reshape( -1, 28, 28, 3)  # 데이터갯수에 -1을 쓰면 reshape시 자동계산됩니다.
# # x_test_reshaped=x_test_norm.reshape( -1, 28, 28, 1)
#
# print("After Reshape - x_train_reshaped shape: {}".format(x_train_reshaped.shape))
# # print("After Reshape - x_test_reshaped shape: {}".format(x_test_reshaped.shape))


image_dir_path = os.getenv("HOME") + "/PycharmProjects/gawibawibo/datasets/test"
(x_test, y_test) = load_data(image_dir_path)
x_test_norm = x_test / 255.0  # 입력은 0~1 사이의 값으로 정규화


test_loss, test_accuracy = model.evaluate(x_test,y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))