print('importing matplotlib')
import matplotlib.pyplot as plt
print('importing seaborn')
import seaborn as sns

print('importing keras')
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
print('importing sklearn')
from sklearn.metrics import classification_report,confusion_matrix

print('importing tensorflow')
import tensorflow as tf

import cv2
import os

print('importing numpy')
import numpy as np

from pathlib import Path
from PIL import Image

import time
import pickle

PATH = Path(__file__).parent

labels = ['on', 'off']
img_size = 224
def load_images(train):
    data = []
    for class_num, label in enumerate(labels):
        image_paths = os.path.join(PATH, label, f'{"train" if train else "test"}')
        for image in os.listdir(image_paths):
            if not image.endswith('.png'):
                print(f'Loading images - ignoring file {image}')
                continue
            try:
                img_arr = cv2.imread(os.path.join(image_paths, image))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)


    return np.array(data, dtype="object")

train = load_images(train=True)
val = load_images(train=False)


x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

opt = Adam(lr=0.000001)
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

model.fit(x_train,y_train,epochs = 500 , validation_data = (x_val, y_val))

model.save('./finalized_model')

