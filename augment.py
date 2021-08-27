# Source code for augmenting the datasets to your computer directory.

from keras.preprocessing.image import ImageDataGenerator
from skimage import io
datagen = ImageDataGenerator(
            rotation_range=5,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.1)


# Your image directories and saving directory.

image_directory = '/Research/COdes/CRC_classification/CRC_data_B/tr_va_ts/2 class/train'
save_directory = '/Research/COdes/CRC_classification/CRC_data_B/tr_va_ts/2 class/augment/stroma'
# SIZE = 150
# dataset = []
#
# my_images = os.listdir(image_directory)
# for i, image_name in enumerate(my_images):
#     if (image_name.split('.')[1] == 'jpg'):
#         image = io.imread(image_directory + image_name)
#         image = Image.fromarray(image, 'RGB')
#         image = image.resize((SIZE, SIZE))
#         dataset.append(np.array(image))
#
# x = np.array(dataset)

i = 0
for batch in datagen.flow_from_directory(directory=image_directory,
                                         batch_size=64,
                                         color_mode="rgb",
                                         save_to_dir=save_directory,
                                         save_prefix='aug',
                                         save_format='png'):
    i += 1
    if i > 31:
        break