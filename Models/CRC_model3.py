# The whole code was referenced from TensorFlow website: https://www.tensorflow.org/tutorials/images/transfer_learning
# This code was used to implement the conference paper submitted for ICTC-2021.

# Note: This is a basic code for CRC image classification tested with just accuracy metrics.
# For multiple class classification and to show the results in different metrics use different code (CRC_model4)

import matplotlib.pyplot as plt
import tensorflow as tf
# from sklearn.metrics import accuracy_score, f1_score, precision_score, confusion_matrix


BATCH_SIZE = 32
IMG_SIZE = (150, 150)

img_height = 150
img_width = 150
batch_size = 32

# Specify your training and testing directory ####
data_dir = '/Research/COdes/CRC_classification/CRC_data_B/2 class/train'
test_path = '/Research/COdes/CRC_classification/CRC_data_B/2 class/test'

# Preparing your dataset with keras.preprocessing
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_batches = tf.data.experimental.cardinality(validation_dataset)
test_dataset = validation_dataset.take(val_batches // 5)
validation_dataset = validation_dataset.skip(val_batches // 5)

print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
print('Number of test batches: %d' % tf.data.experimental.cardinality(test_dataset))

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
  tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
])

preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input
rescale = tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)

# Create the base model from the pre-trained model MobileNet V2.
# To use other pretrained models you can change the model name in the base model.

IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.InceptionResNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

image_batch, label_batch = next(iter(train_dataset))
# print(image_batch)
# print(label_batch)

feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
# Let's take a look at the base model architecture
base_model.summary()

# Creating a classification head to attach on top of the pretrained model as above.
# You can use max pooling or global pooling according to your need.


# global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
global_maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
# cat_layer = tf.concat([global_average_layer, global_maxpool_layer], axis = 3)
feature_batch_average = global_maxpool_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(1)
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)


# Combining the whole model
inputs = tf.keras.Input(shape=(150, 150, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_maxpool_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)


# Specifying training parameters
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.summary()
len(model.trainable_variables)

initial_epochs = 100

loss0, accuracy0 = model.evaluate(validation_dataset)

print("initial loss: {:.2f}".format(loss0))
print("initial accuracy: {:.2f}".format(accuracy0))

# Fit the model
history = model.fit(train_dataset,
                    epochs=initial_epochs,
                    validation_data=validation_dataset)
model.save('inceptionresnetv2.h5')  # save your model in .h5 file

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

# Plotting the learning curves
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


# Evaluating on test dataset to obtain the test accuracy
loss, accuracy = model.evaluate(test_dataset)
print('Test accuracy :', accuracy)