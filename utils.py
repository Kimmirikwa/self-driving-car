import os, cv2
import numpy as np
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.preprocessing.image import random_shift, random_rotation, ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasRegressor

from constants import DATA_DIR, SAMPLES_PER_EPOCH, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, BATCH_SIZE, SHIFT_RANGE

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
        Resize the image to the input shape used by the network model i.e 200 by 66
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)


def rgb2yuv(image):
    """
        Convert the image from RGB to YUV
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image, resize_image=True):
    """
        preprocess image before feeding to model
        1. crop the image to remove unnecessary parts
        2. convert from RGB to YUV
        3. convert to the shape expected by the the input of the model
    """
    image = crop(image)
    image = rgb2yuv(image)
    image = resize(image)
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
    	angle_change = -0.2
    return preprocess(load_image(data_dir, camera)), steering_angle + angle_change

def do_horizontal_flip():
    flip_image = False
    choice = np.random.choice(2)
    if choice == 0:
        flip_image = True

    return flip_image

def random_shift(image, steering_angle, range_x, range_y):
    """
        Randomly shift the image both vertically and horizontally.
        when we do a horizontal shift, we need to change the steering angle too
        as the horizontal view of the road changes too
    """
    trans_x = range_x * (np.random.rand() - 0.5)
    trans_y = range_y * (np.random.rand() - 0.5)
    steering_angle += trans_x * 0.002
    trans_m = np.float32([[1, 0, trans_x], [0, 1, trans_y]])
    height, width = image.shape[:2]
    image = cv2.warpAffine(image, trans_m, (width, height))
    return image, steering_angle

def argument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
    '''
        Image augmentation.
        randomly one of the images from 'center', 'left' and right
        we the apply random transformations on the selected image.
        for transformations that do not require a change in the steering angle we use 'ImageDataGenerator'
        otherwise a custom transformer is used
    '''
    image, steering_angle = choose_image(data_dir, center, left, right, steering_angle) # select one image

    image, steering_angle = random_shift(image, steering_angle, range_x, range_y)  # shifting horizontally needs some change in the steering angle

    # if we are doing horizontal flip, we negate the steering angle
    horizontal_flip = do_horizontal_flip()
    if horizontal_flip:
        steering_angle = -steering_angle

    datagen = ImageDataGenerator(
        rotation_range=40,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=horizontal_flip,
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