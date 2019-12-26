import os, cv2
import numpy as np
import matplotlib.image as mpimg
from keras.preprocessing.image import random_shift, random_rotation

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

def random_flip(image, steering_angle):
    """
    Randomly flipt the image left <-> right, and adjust the steering angle.
    """
    if np.random.rand() < 0.5:
        image = cv2.flip(image, 1)
        steering_angle = -steering_angle
    return image, steering_angle

def shift_image(image):
	image = random_shift(
		image,
		0.1,
		0.1,
		row_axis=1,
		col_axis=2,
		channel_axis=0,
		fill_mode='nearest',
		cval=0.0,
		interpolation_order=1)

	return image

def rotate_image(image):
	image = random_rotation(
	    image,
	    20,
	    row_axis=1,
	    col_axis=2,
	    channel_axis=0,
	    fill_mode='nearest',
	    cval=0.0,
	    interpolation_order=1
	)

	return image

def random_shadow(image):
    """
    Generates and adds random shadow
    """
    # (x1, y1) and (x2, y2) forms a line
    # xm, ym gives all the locations of the image
    x1, y1 = IMAGE_WIDTH * np.random.rand(), 0
    x2, y2 = IMAGE_WIDTH * np.random.rand(), IMAGE_HEIGHT
    xm, ym = np.mgrid[0:IMAGE_HEIGHT, 0:IMAGE_WIDTH]

    # mathematically speaking, we want to set 1 below the line and zero otherwise
    # Our coordinate is up side down.  So, the above the line: 
    # (ym-y1)/(xm-x1) > (y2-y1)/(x2-x1)
    # as x2 == x1 causes zero-division problem, we'll write it in the below form:
    # (ym-y1)*(x2-x1) - (y2-y1)*(xm-x1) > 0
    mask = np.zeros_like(image[:, :, 1])
    mask[(ym - y1) * (x2 - x1) - (y2 - y1) * (xm - x1) > 0] = 1

    # choose which side should have shadow and adjust saturation
    cond = mask == np.random.randint(2)
    s_ratio = np.random.uniform(low=0.2, high=0.5)

    # adjust Saturation in HLS(Hue, Light, Saturation)
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls[:, :, 1][cond] = hls[:, :, 1][cond] * s_ratio
    return cv2.cvtColor(hls, cv2.COLOR_HLS2RGB)


def random_brightness(image):
    """
    Randomly adjust brightness of the image.
    """
    # HSV (Hue, Saturation, Value) is also called HSB ('B' for Brightness).
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    ratio = 1.0 + 0.4 * (np.random.rand() - 0.5)
    hsv[:,:,2] =  hsv[:,:,2] * ratio
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

def argument(data_dir, center, left, right, steering_angle, range_x=100, range_y=10):
	# select one of the three images
	image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
	image, steering_angle = random_flip(image, steering_angle)
	image = shift_image(image)
	image = rotate_image(image)
	image = random_shadow(image)
	image = random_brightness(image)

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