import os, cv2
import numpy as np
import matplotlib.image as mpimg
from keras.preprocessing.image import random_shift, random_rotation, ImageDataGenerator

from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, BATCH_SIZE

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    return image[60:-25, :, :] # remove the sky and the car front


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right
    the steering is adjusted when left or right camera is selected
    """
    choice = np.random.choice(3)
    camera = center
    angle_change = 0
    if choice == 0:
    	camera = left
    	angle_change = 0.2
    elif choice == 1:
    	camera = right
    	angle_change = -2
    return preprocess(load_image(data_dir, camera)), steering_angle + angle_change

def argument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)

    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    images = datagen.flow(image.reshape((1,) + image.shape), batch_size=1)
    image = images[0]

    return image, steering_angle

def data_batch_generator(data_dir, image_paths, steering_angles, batch_size=BATCH_SIZE, is_training=True):
	'''
		gets image paths and uses them to read images. the images are then mapped to their steering angles.
		we argument 3 images into one during training
		params:
			- data_dir : the source of the data file
			- image_paths: the columns with path to images
			- steering_angles: column with steering angles
			- batch_size: the number of examples per batch
			- is_training: flag to indicate if we are training
	'''
	images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
	steers = np.empty(batch_size)

	while True:
		i = 0
		# randomly rearange the indices of the data i.e from the rows
		index_perm = np.random.permutation(image_paths.shape[0])

		for index in index_perm:
			# unpack the paths for images, we need to use the path to read the images
			center, left, right = image_paths.iloc[index]
			steering_angle = steering_angles.iloc[index]  # this is ready to be used in the model

			# attempt to argument the image when training
			if is_training and np.random.rand() < 0.7:
				image, steering_angle = argument(data_dir, center, left, right, steering_angle)
			else:
				image = preprocess(load_image(data_dir, center))

			images[i] = image
			steers[i] = steering_angle

			i += 1
			if i == batch_size:
				break

		yield images, steers