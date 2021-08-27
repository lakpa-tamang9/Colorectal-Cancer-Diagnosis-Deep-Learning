# -*- coding: utf-8 -*-

# This source code loads the pretrained model that was trained and saved earlier in .h5 file. Using this model, we can evaluate the performance on the testing dataset.
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import matplotlib.pyplot as plt


# Training and testing directory
train_path = '/Research/COdes/CRC_classification/CRC_data_B/2 class/train'
test_path = '/Research/COdes/CRC_classification/CRC_data_B/2 class/test'

random_seed = 101

train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)
test_datagen = ImageDataGenerator(rescale=1./255)
    #rescale=1./255,
    #shear_range=0.2,
    #zoom_range=0.2,
    #horizontal_flip=True)


# The below snippet should be run if the program throws GPU error. If not just run as it is.
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)

target_size=(150,150,3)

train_generator = train_datagen.flow_from_directory(train_path,
    target_size=target_size[:-1],
    batch_size=32,
    class_mode='binary',
    subset='training',
    seed=random_seed)

valid_generator = train_datagen.flow_from_directory(train_path,
    target_size=target_size[:-1],
    batch_size=32,
    class_mode='binary',
    subset='validation',
    seed=random_seed)

test_generator = test_datagen.flow_from_directory(test_path,
  target_size=target_size[:-1],
   batch_size=32,
   class_mode='binary',color_mode="grayscale",
    shuffle=False)

n_classes = len(set(train_generator.classes))
#momentum = 0.98
print(n_classes)

# model = applications.DenseNet121(weights='imagenet', include_top=False)
# print('Model loaded.')
# model.summary()


preprocess_input = tf.keras.applications.inception_resnet_v2.preprocess_input

global_maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
dense1 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')
dense2 = tf.keras.layers.Dense(256, activation='relu', kernel_initializer='he_normal')
prediction_layer = tf.keras.layers.Dense(1, activation='softmax')
base_model = tf.keras.applications.InceptionResNetV2(input_shape=(150, 150, 3),
                                               include_top=False,
                                               weights='imagenet')

inputs = tf.keras.Input(shape=(150, 150, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_maxpool_layer(x)
x = tf.keras.layers.Dropout(0.3)(x)
x = dense1(x)
# x = tf.keras.layers.Dropout(0.3)(x)
# x = dense2(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

epochs = 10
history = model.fit_generator(generator=train_generator, validation_data=valid_generator, epochs=epochs)

model.save('crcnew_model.h5')

# score = model.evaluate_generator(test_generator)
#
# predictions = model.predict_generator(test_generator)
# labels = test_generator.labels
# classes = test_generator.class_indices.keys()
# classes = list(classes)
#
# corrections = np.argmax(predictions,axis=-1) == labels
# corrections = np.mean(corrections)
#
# np.savez_compressed('./results.npz', predictions=predictions, labels=labels, classes=classes)

print(history.history.keys())
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()





