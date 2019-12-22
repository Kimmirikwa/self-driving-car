import os
import numpy as np
import matplotlib.image as mpimg

from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, BATCH_SIZE

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))

def choose_image(data_dir, center, left, right, steering_angle):
    """
    Randomly choose an image from the center, left or right, and adjust
    the steering angle.
    """
    choice = np.random.choice(3)
    if choice == 0:
        return load_image(data_dir, left), steering_angle + 0.2
    elif choice == 1:
        return load_image(data_dir, right), steering_angle - 0.2
    return load_image(data_dir, center), steering_angle

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
			if is_training:
				image, steering_angle = choose_image(data_dir, center, left, right, steering_angle)
			else:
				image = load_image(data_dir, center)

			images[i] = image
			steers[i] = steering_angle

			i += 1
			if i == batch_size:
				break

		yield images, steers