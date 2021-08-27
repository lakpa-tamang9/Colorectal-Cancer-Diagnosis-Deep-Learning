# Source code using stratified K-fold


import numpy as np  # linear algebra
from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix
from sklearn.model_selection import KFold
from PIL import Image
import random
# Dependencies
import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
# CNN
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.normalization import BatchNormalization
# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession
import warnings
import os
import shutil
from PIL import ImageFile
from keras import applications
import matplotlib.pyplot as plt

random_seed = 101
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)

# gpus = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(gpus[0], True)

warnings.simplefilter('error', Image.DecompressionBombWarning)
ImageFile.LOAD_TRUNCATED_IMAGES = True
from PIL import Image

devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

Image.MAX_IMAGE_PIXELS = 1000000000

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)

datasetFolderName = 'CRC data'
MODEL_FILENAME = "model_cv.h5"
sourceFiles = []
classLabels = ['tumor', 'stroma']


def transferBetweenFolders(source, dest, splitRate):
    global sourceFiles
    sourceFiles = os.listdir(source)
    if (len(sourceFiles) != 0):
        transferFileNumbers = int(len(sourceFiles) * splitRate)
        transferIndex = random.sample(range(0, len(sourceFiles)), transferFileNumbers)
        for eachIndex in transferIndex:
            shutil.move(source + str(sourceFiles[eachIndex]), dest + str(sourceFiles[eachIndex]))
    else:
        print("No file moved. Source empty!")


def transferAllClassBetweenFolders(source, dest, splitRate):
    for label in classLabels:
        transferBetweenFolders(datasetFolderName + '/' + source + '/' + label + '/',
                               datasetFolderName + '/' + dest + '/' + label + '/',
                               splitRate)


# First, check if test folder is empty or not, if not transfer all existing files to train
transferAllClassBetweenFolders('test', 'train', 1.0)
# Now, split some part of train data into the test folders.
transferAllClassBetweenFolders('train', 'test', 0.20)

X = []
Y = []


def prepareNameWithLabels(folderName):
    sourceFiles = os.listdir(datasetFolderName + '/train/' + folderName)
    for val in sourceFiles:
        X.append(val)
        if (folderName == classLabels[0]):
            Y.append(0)
        else:
            Y.append(1)
        # else:
        #     Y.append(2)


# Organize file names and class labels in X and Y variables
prepareNameWithLabels(classLabels[0])
prepareNameWithLabels(classLabels[1])
# prepareNameWithLabels(classLabels[2])

X = np.asarray(X)
Y = np.asarray(Y)

# learning rate
batch_size = 32
epoch = 10
activationFunction = 'relu'

preprocess_input = tf.keras.applications.densenet.preprocess_input
global_maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
prediction_layer = tf.keras.layers.Dense(1)
base_model = tf.keras.applications.DenseNet121(input_shape=(150, 150, 3),
                                               include_top=False,
                                               weights='imagenet')

inputs = tf.keras.Input(shape=(150, 150, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_maxpool_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# def getModel():
#     model = Sequential()
#     model.add(Conv2D(64, (3, 3), padding='same', activation=activationFunction, input_shape=(img_rows, img_cols, 3)))
#     model.add(Conv2D(64, (3, 3), activation=activationFunction))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(32, (3, 3), padding='same', activation=activationFunction))
#     model.add(Conv2D(32, (3, 3), activation=activationFunction))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Conv2D(16, (3, 3), padding='same', activation=activationFunction))
#     model.add(Conv2D(16, (3, 3), activation=activationFunction))
#     model.add(MaxPooling2D(pool_size=(2, 2)))
#     model.add(Dropout(0.25))
#
#     model.add(Flatten())
#     model.add(Dense(64, activation=activationFunction))  # we can drop
#     model.add(Dropout(0.1))  # this layers
#     model.add(Dense(32, activation=activationFunction))
#     model.add(Dropout(0.1))
#     model.add(Dense(16, activation=activationFunction))
#     model.add(Dropout(0.1))
#     model.add(Dense(1, activation='softmax'))
#
#     model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#
#     return model


def my_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    f1Score = f1_score(y_true, y_pred, average='weighted')
    print("Accuracy  : {}".format(accuracy))
    print("Precision : {}".format(precision))
    print("f1Score : {}".format(f1Score))
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    return accuracy, precision, f1Score


# input image dimensions
img_rows, img_cols = 150, 150

train_path = datasetFolderName + '/train/'
validation_path = datasetFolderName + '/validation/'
test_path = datasetFolderName + '/test/'
# model = getModel()

# ===============Stratified K-Fold======================
skf = KFold(n_splits=3, shuffle=True)
skf.get_n_splits(X, Y)
foldNum = 0
for train_index, val_index in skf.split(X, Y):
    # First cut all images from validation to train (if any exists)
    transferAllClassBetweenFolders('validation', 'train', 1.0)
    foldNum += 1
    print("Results for fold", foldNum)
    X_train, X_val = X[train_index], X[val_index]
    Y_train, Y_val = Y[train_index], Y[val_index]
    # Move validation images of this fold from train folder to the validation folder
    for eachIndex in range(len(X_val)):
        classLabel = ''
        if (Y_val[eachIndex] == 0):
            classLabel = classLabels[0]
        else:
            classLabel = classLabels[1]
            # Then, copy the validation images to the validation folder
        shutil.move(datasetFolderName + '/train/' + classLabel + '/' + X_val[eachIndex],
                    datasetFolderName + '/validation/' + classLabel + '/' + X_val[eachIndex])

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        zoom_range=0.20,
        horizontal_flip=True,
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        vertical_flip=True,
        fill_mode="reflect"
    )
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    # test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Start ImageClassification Model
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='binary',
        subset='training')

    validation_generator = validation_datagen.flow_from_directory(
        validation_path,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode=None,  # only data, no labels
        shuffle=False)

    # fit model
    history = model.fit_generator(train_generator,
                                  epochs=epoch)

    predictions = model.predict_generator(validation_generator, verbose=1)
    yPredictions = np.argmax(predictions, axis=1)
    true_classes = validation_generator.classes
    # evaluate validation performance
    print("***Performance on Validation data***")
    valAcc, valPrec, valFScore = my_metrics(true_classes, yPredictions)

# =============TESTING=============
print("==============TEST RESULTS============")
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode=None,
    shuffle=False)
predictions = model.predict(test_generator, verbose=1)
yPredictions = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

testAcc, testPrec, testFScore = my_metrics(true_classes, yPredictions)
model.save(MODEL_FILENAME)