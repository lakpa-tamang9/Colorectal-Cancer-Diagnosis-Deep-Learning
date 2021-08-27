import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from keras import applications
import os
import cv2
from PIL import Image
import numpy as np
from tensorflow import keras


train_dir = '/Research/COdes/CRC_classification/Data CRC/train/'
test_path = '/Research/COdes/CRC_classification/Data CRC/test/'

SIZE = 150
dataset = []
label = []

stroma_images = os.listdir(train_dir + 'stroma/')
for i, image_name in enumerate(stroma_images):
    image = Image.open(train_dir + 'stroma/' + image_name)
    # image = Image.fromarray(image)
    image = image.resize((SIZE, SIZE))
    dataset.append(np.array(image))
    label.append(0)

tumor_images = os.listdir(train_dir + 'tumor/')
for i, image_name in enumerate(tumor_images):
    image = Image.open(train_dir + 'tumor/' + image_name)
    image = image.resize((SIZE, SIZE))
    dataset.append(np.array(image))
    label.append(1)

dataset = np.array(dataset)
label = np.array(label)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(dataset, label, test_size=0.3, random_state=None)

from keras.utils import normalize
X_train = normalize(X_train, axis=1)
X_test = normalize(X_test, axis=1)

INPUT_SHAPE = (SIZE, SIZE, 3)

preprocess_input = applications.inception_resnet_v2.preprocess_input
data_augmentation = keras.Sequential([
  keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  keras.layers.experimental.preprocessing.RandomFlip('vertical'),
  keras.layers.experimental.preprocessing.RandomRotation(0.1),
])
# Create the base model from the pre-trained model MobileNet V2
base_model = keras.applications.InceptionResNetV2(input_shape=INPUT_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

base_model.trainable = False
# Let's take a look at the base model architecture
base_model.summary()

# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
global_maxpool_layer = keras.layers.GlobalMaxPooling2D()


prediction_layer = keras.layers.Dense(1, activation='sigmoid')


inputs = keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_maxpool_layer(x)
x = keras.layers.Dropout(0.5)(x)
outputs = prediction_layer(x)
model = keras.Model(inputs, outputs)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, y_train, batch_size=32, verbose=1, epochs=200, validation_data=(X_test, y_test), shuffle=False)

model.save('shreeni_model.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()),1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0,1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
plt.show()
