# -*- coding: utf-8 -*-
"""Malaria Detection with CNNs

# **Malaria Detection**

###<b> Mount the Drive
"""

# prompt: mount drive

from google.colab import drive
drive.mount('/content/drive')

pip install Pillow

pip install tensorflow

"""### <b>Loading libraries</b>"""

# Importing libraries for our project
import zipfile

import os

from PIL import Image

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

import seaborn as sns

import tensorflow as tf

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras import backend

from tensorflow.keras.utils import to_categorical

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation, LeakyReLU

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from random import shuffle

# To ignore warnings
import warnings

warnings.filterwarnings('ignore')

# Storing the path of the data file from the Google drive
path = '/content/drive/MyDrive/MIT Python Work/Capstone Project/cell_images.zip'

# Extracting images with zipfile
with zipfile.ZipFile(path, 'r') as zip_ref:

    zip_ref.extractall()

# Storing the path of the extracted "train" folder
train_dir = '/content/cell_images/train'

# Size of image so that each image has the same size
SIZE = 64

# Empty list to store the training images after they are converted to NumPy arrays
train_images = []

# Empty list to store the training labels (0 - uninfected, 1 - parasitized)
train_labels = []

#Creating folders for our Train images broken into Parasitized and Unifected
for folder_name in ['/parasitized/', '/uninfected/']:

    # Path of the folder
    images_path = os.listdir(train_dir + folder_name)

    for i, image_name in enumerate(images_path):

        try:

            # Opening each image using the path of that image
            image = Image.open(train_dir + folder_name + image_name)

            # Resizing each image to (64, 64)
            image = image.resize((SIZE, SIZE))

            # Converting images to arrays and appending that array to the empty list defined above
            train_images.append(np.array(image))

            # Creating labels for parasitized and uninfected images
            if folder_name == '/parasitized/':

                train_labels.append(1)

            else:

                train_labels.append(0)

        except Exception:

            pass

# Converting lists to arrays
train_images = np.array(train_images)

train_labels = np.array(train_labels)

# Storing the path of the extracted "test" folder
test_dir = '/content/cell_images/test'

# Size of image so that each image has the same size (it must be same as the train image size)
SIZE = 64

# Empty list to store the testing images after they are converted to NumPy arrays
test_images = []

# Empty list to store the testing labels (0 - uninfected, 1 - parasitized)
test_labels = []

#Creating folders for our Train images broken into Parasitized and Unifected
for folder_name in ['/parasitized/', '/uninfected/']:

    # Path of the folder
    images_path = os.listdir(test_dir + folder_name)

    for i, image_name in enumerate(images_path):

        try:
            # Opening each image using the path of that image
            image = Image.open(test_dir + folder_name + image_name)

            # Resizing each image to (64, 64)
            image = image.resize((SIZE, SIZE))

            # Converting images to arrays and appending that array to the empty list defined above
            test_images.append(np.array(image))

            # Creating labels for parasitized and uninfected images
            if folder_name == '/parasitized/':

                test_labels.append(1)

            else:

                test_labels.append(0)

        except Exception:

            pass

# Converting lists to arrays
test_images = np.array(test_images)

test_labels = np.array(test_labels)

#Prints shape of our images
print(train_images.shape)
print(test_images.shape)

#Print shape of our labels
print(train_labels.shape)
print(test_labels.shape)

"""(99832, 2, 2, 2)
(10400, 2, 2, 2)
"""

#Print the min and max pixel values for images
print(train_images.min())
print(train_images.max())
print(test_images.min())
print(test_images.max())

#Creates variables that adds the number of images per diagnosis and by train/test folder
num_infected_train = np.sum(train_labels == 1)
num_infected_test = np.sum(test_labels == 1)

num_uninfected_train = np.sum(train_labels == 0)
num_uninfected_test = np.sum(test_labels == 0)

#Number of images showing infected/parasitized blood cells
print(num_infected_test + num_infected_train)

#Number of uninfected blood cells
print(num_uninfected_test + num_uninfected_train)

#Normalizes our images to 0-1 and changes type to float32
train_images = (train_images/255).astype('float32')
test_images = (test_images/255).astype('float32')

#Sets random seed
np.random.seed(42)

#Plots 16 random images from our Train Image folder
plt.figure(1, figsize = (10 , 10))

for n in range(1, 17):

    plt.subplot(4, 4, n)

    index = int(np.random.randint(0, train_images.shape[0], 1))

    plt.imshow(train_images[index])

    plt.axis('off')

#Sets random seed
np.random.seed(42)

#Plots 16 random images from our Train Image folder

plt.figure(1, figsize = (10 , 10))

for n in range(1, 17):

    plt.subplot(4, 4, n)

    index = int(np.random.randint(0, train_images.shape[0], 1))

    if train_labels[index] == 1:

        plt.title('Infected (parasitized)')

    else:
        plt.title('Uninfected')

    plt.imshow(train_images[index])

    plt.axis('off')

# Function to find the mean
def find_mean_img(full_mat, title):

    # Calculate the average
    mean_img = np.mean(full_mat, axis = 0)[0]

    # Reshape it back to a matrix
    plt.imshow(mean_img)

    plt.title(f'Average {title}')

    plt.axis('off')

    plt.show()

    return mean_img

#Plots the mean image for parasitized images (labeled as 1)
parasitized_data = []  # Create a list to store the parasitized data

for img, label in zip(train_images, train_labels):

        if label == 1:

              parasitized_data.append(img)

parasitized_array = np.array(parasitized_data)
parasitized_mean = find_mean_img(parasitized_array, 'Infected')

#Plots the mean image for parasitized images (labeled as 0)
uninfected_data = []  # Create a list to store the uninfected data

for img, label in zip(train_images, train_labels):

        if label == 0:

              uninfected_data.append(img)

uninfected_array = np.array(uninfected_data)
parasitized_mean = find_mean_img(uninfected_array, 'Uninfected')

#Converting Train Data to HSV
import cv2

gfx=[]   # to hold the HSV image array

for i in np.arange(0, 100, 1):

  a = tf.image.rgb_to_hsv(train_images[i])

  gfx.append(a)

gfx = np.array(gfx)

viewimage = np.random.randint(1, 100, 10)

fig, ax = plt.subplots(1, 10, figsize = (10, 10))

for t, i in zip(range(10), viewimage):

  if train_labels[index] == 1:

        plt.title('Infected')

    else:
        plt.title('Uninfected')

  ax[t].imshow(gfx[i])

  ax[t].set_axis_off()

  fig.tight_layout()

#Converting Test Data to HSV
import cv2

gfx=[]   # to hold the HSV image array

for i in np.arange(0, 100, 1):

  a = tf.image.rgb_to_hsv(test_images[i])

  gfx.append(a)

gfx = np.array(gfx)

viewimage = np.random.randint(1, 100, 10)

fig, ax = plt.subplots(1, 10, figsize = (10, 10))

for t, i in zip(range(10), viewimage):


  ax[t].imshow(gfx[i])

  ax[t].set_axis_off()

  fig.tight_layout()

#Converting Train Images with Gaussian Blurring
gbx = []  # To hold the blurred images

for i in np.arange(0, 100, 1):

  b = cv2.GaussianBlur(train_images[i], (5, 5), 0)

  gbx.append(b)

gbx = np.array(gbx)

viewimage = np.random.randint(1, 100, 10)

fig, ax = plt.subplots(1, 10, figsize = (10, 10))

for t, i in zip(range(10), viewimage):

  ax[t].imshow(gbx[i])

  ax[t].set_axis_off()

  fig.tight_layout()

#Converting Test Images with Gaussian Blurring
gbx = []  # To hold the blurred images

#iterates through our images and converts them to include Gaussian Blurring
for i in np.arange(0, 100, 1):

  b = cv2.GaussianBlur(test_images[i], (5, 5), 0)

  gbx.append(b)

gbx = np.array(gbx)

#plots 10 Gaussian Blurred images from our Test Images
viewimage = np.random.randint(1, 100, 10)

fig, ax = plt.subplots(1, 10, figsize = (10, 10))

for t, i in zip(range(10), viewimage):

  ax[t].imshow(gbx[i])

  ax[t].set_axis_off()

  fig.tight_layout()

# Clearing backend
backend.clear_session()

# Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)

from keras.utils import to_categorical

# Encoding Train Labels
train_labels = tf.keras.utils.to_categorical(train_labels, 2)

# Encoding Test Labels
test_labels = tf.keras.utils.to_categorical(test_labels, 2)

#Reprinting shape to confirm that our image shape is in 64, 64, 3 shape for our models
print(test_images.shape)

# Initiating the model
model = Sequential()

# Adding the first layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu", input_shape = (64, 64, 3)))

# Adding MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adding Dropout
model.add(Dropout(0.2))

# Adding the second layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

# Adding MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adding dropout
model.add(Dropout(0.2))

# Adding the third layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

# Adding Maxpooling
model.add(MaxPooling2D(pool_size = 2))

# Adding Dropout
model.add(Dropout(0.2))

# Flattens our output to a 1D vector
model.add(Flatten())

# Adding a Dense layer with 512 neurons and relu activition
model.add(Dense(512, activation = "relu"))

# Adding Dropout
model.add(Dropout(0.4))

# Adding a Dense layer with 2 neurons and softmax activition
model.add(Dense(2, activation = "softmax"))

# Compiles the model with binary crossentropy as we have two possible classes (Parasitzed or Uninfected)
# Using adam as an optimizer to improve performance
# Analyzing our model based on accuracy
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Stop running our model up to 4 epochs after the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=4)

history = model.fit(
    train_images, train_labels,
    epochs=20,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stop, reduce_lr]
)

accuracy = model.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test Accuracy:-', accuracy[1])

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (8, 5))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()

# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (7, 7))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "Train Accuracy", color = '#8e24aa')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "Val Accuracy", color = '#CCFF00')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper left")

plot_accuracy(history)

# Clearing backend
backend.clear_session()

# Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)

# Initiating the model
model = Sequential()

# Adding the first layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu", input_shape = (64, 64, 3)))

# Adding MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adding Dropout
model.add(Dropout(0.2))

# Adding second layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

# Adding MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adding Dropout
model.add(Dropout(0.2))

# Adding Third layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

# Adding Maxpooling
model.add(MaxPooling2D(pool_size = 2))

# Adding the third layer
model.add(Conv2D(filters = 32, kernel_size = (3, 3), padding = 'same'))

# Adding LeakyReLu activation function
model.add(LeakyReLU(0.1))

# Adding the fourth layer
model.add(Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same'))

# Adding LeakyReLu activation function
model.add(LeakyReLU(0.1))

# Adding max pooling to reduce the size of the output of fourth convolutional layer
model.add(MaxPooling2D(pool_size = (2, 2)))

#Adding Dropout
model.add(Dropout(0.2))

#Flatten to 1D
model.add(Flatten())

# Adding a Dense layer with 512 neurons and relu activition
model.add(Dense(512, activation = "relu"))

# Adding Dropout
model.add(Dropout(0.4))

# Adding a Dense layer with 2 neurons and relu activition
model.add(Dense(2, activation = "relu"))

# Compiles the model with binary crossentropy as we have two possible classes (Parasitzed or Uninfected)
# Using adam as an optimizer to improve performance
# Analyzing our model based on accuracy
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Stop running our model up to 2 epochs after the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

# Fitting our model with train images and train labels
history = model.fit(train_images, train_labels,
                    batch_size = 32,
                    callbacks = [early_stop, reduce_lr],
                    validation_split = 0.2,
                    epochs = 20,
                    verbose = 1)

accuracy = model.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test Accuracy:-', accuracy[1])

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (8, 5))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()

# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (7, 7))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "Train Accuracy", color = '#8e24aa')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "Val Accuracy", color = '#CCFF00')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper left")

plot_accuracy(history)

# Clearing backend
backend.clear_session()

# Fixing the seed for random number generators so that we can ensure we receive the same output everytime
np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)

# Initiating the model
model = Sequential()

# Adding first layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", input_shape = (64, 64, 3)))

# Adding LeakyRelu activation function
model.add(LeakyReLU(0.1))

# Adding MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adding Dropout
model.add(Dropout(0.2))

# Adding the second layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same"))

# Adding LeakyRelu activation function
model.add(LeakyReLU(0.1))

# Adding MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adding Dropout
model.add(Dropout(0.2))

#Adding the third layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same"))

# Adding LeakyRelu activation function
model.add(LeakyReLU(0.1))

# Adding MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adding Batch Normalization
model.add(BatchNormalization())

# Adding Dropout
model.add(Dropout(0.2))

# Flatten to 1D matrix
model.add(Flatten())

# Adding Dense
model.add(Dense(512))

# Adding Dropout
model.add(Dropout(0.4))

# Adding Dense
model.add(Dense(2))

# Compiles the model with binary crossentropy as we have two possible classes (Parasitzed or Uninfected)
# Using adam as an optimizer to improve performance
# Analyzing our model based on accuracy
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Stop running our model up to 2 epochs after the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

# Fitting our model with train images and train labels]
history = model.fit(train_images, train_labels,
                    batch_size = 32,
                    callbacks = [early_stop, reduce_lr],
                    validation_split = 0.2,
                    epochs = 20,
                    verbose = 1)

accuracy = model.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test Accuracy:-', accuracy[1])

# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (7, 7))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "Train Accuracy", color = '#8e24aa')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "Val Accuracy", color = '#CCFF00')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper left")

plot_accuracy(history)

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (8, 5))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()

backend.clear_session() # Clearing backend for new model

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size = 0.2, random_state = 42)

from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Using ImageDataGenerator to generate images
train_datagen = ImageDataGenerator(horizontal_flip = True,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.15,
                                    zoom_range = 0.3, rotation_range = 30)


val_datagen  = ImageDataGenerator()

# Flowing training images using train_datagen generator
train_generator = train_datagen.flow(x = X_train , y = y_train, batch_size = 64, seed = 42, shuffle = True)


# Flowing validation images using val_datagen generator
val_generator =  val_datagen.flow(x = X_val, y = y_val , batch_size = 64, seed = 42, shuffle = True)

# Creating an iterable for images and labels from the training data
images, labels = next(train_generator)

# Plotting 16 images from the training data
fig, axes = plt.subplots(4, 4, figsize = (10, 10))

fig.set_size_inches(10, 10)
for (image, label, ax) in zip(images, labels, axes.flatten()):

    ax.imshow(image)

    if label[1] == 1:

        ax.set_title('parasitized')

    else:

        ax.set_title('uninfected')

    ax.axis('off')

# Initiates model
model = Sequential()

# Adds the first layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu", input_shape = (64, 64, 3)))

# Adds MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adds a droput to the output of the first layer
model.add(Dropout(0.2))

# Adds the second layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

# Adds MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adds a droput to the output of the second layer
model.add(Dropout(0.2))

# Adds the third layer
model.add(Conv2D(filters = 32, kernel_size = 2, padding = "same", activation = "relu"))

# Adds MaxPooling
model.add(MaxPooling2D(pool_size = 2))

# Adds a droput to the output of the third layer
model.add(Dropout(0.2))

#Flattens output to 1D matrix
model.add(Flatten())

# Adding a Dense layer with 512 neurons and relu activition
model.add(Dense(512, activation = "relu"))

#Adds a droput
model.add(Dropout(0.4))

# Adding a Dense layer with 2 neurons and softmax activition
model.add(Dense(2, activation = "softmax"))

# Compiles the model with binary crossentropy as we have two possible classes (Parasitzed or Uninfected)
# Using adam as an optimizer to improve performance
# Analyzing our model based on accuracy
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Stop running our model up to 2 epochs after the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2)

history = model.fit(train_generator,
                                  validation_data = val_generator,
                                  batch_size = 32, callbacks = [early_stop, reduce_lr],
                                  epochs = 20, verbose = 1)

accuracy = model.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test Accuracy:-', accuracy[1])

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (8, 5))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()

# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (7, 7))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "Train Accuracy", color = '#8e24aa')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "Val Accuracy", color = '#CCFF00')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper left")

plot_accuracy(history)

# Clearing backend
from tensorflow.keras import backend

backend.clear_session()

# Fixing the seed for random number generators
np.random.seed(42)

import random

random.seed(42)

tf.random.set_seed(42)

from tensorflow.keras.applications.vgg16 import VGG16

from tensorflow.keras import Model

from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout

vgg = VGG16(include_top = False, weights = 'imagenet', input_shape = (64, 64, 3))

vgg.summary()

transfer_layer = vgg.get_layer('block5_pool')

vgg.trainable = False

x = GlobalAveragePooling2D()(vgg.get_layer('block5_pool').output)
x = Dense(128, activation='relu')(x)
x = Dropout(0.3)(x)
pred = Dense(2, activation='softmax')(x)

# Compiles the model with binary crossentropy as we have two possible classes (Parasitzed or Uninfected)
# Using adam as an optimizer to improve performance
# Analyzing our model based on accuracy
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Stop running our model up to 2 epochs after the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=1e-6)

history = model.fit(train_generator,
                                  validation_data = val_generator,
                                  batch_size = 32, callbacks = [early_stop, reduce_lr],
                                  epochs = 20, verbose = 1)

accuracy = model.evaluate(test_images, test_labels, verbose = 1)
print('\n', 'Test Accuracy:-', accuracy[1])

from sklearn.metrics import classification_report

from sklearn.metrics import confusion_matrix

pred = model.predict(test_images)

pred = np.argmax(pred, axis = 1)

y_true = np.argmax(test_labels, axis = 1)

# Printing the classification report
print(classification_report(y_true, pred))

# Plotting the heatmap using confusion matrix
cm = confusion_matrix(y_true, pred)

plt.figure(figsize = (8, 5))

sns.heatmap(cm, annot = True,  fmt = '.0f', xticklabels = ['Uninfected', 'Parasitized'], yticklabels = ['Uninfected', 'Parasitized'])

plt.ylabel('Actual')

plt.xlabel('Predicted')

plt.show()

# Function to plot train and validation accuracy
def plot_accuracy(history):

    N = len(history.history["accuracy"])

    plt.figure(figsize = (7, 7))

    plt.plot(np.arange(0, N), history.history["accuracy"], label = "Train Accuracy", color = '#8e24aa')

    plt.plot(np.arange(0, N), history.history["val_accuracy"], label = "Val Accuracy", color = '#CCFF00')

    plt.title("Accuracy vs Epoch")

    plt.xlabel("Epochs")

    plt.ylabel("Accuracy")

    plt.legend(loc="upper left")

plot_accuracy(history)
