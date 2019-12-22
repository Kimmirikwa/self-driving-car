import numpy as np

from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS, BATCH_SIZE

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
			center, left, right = image_paths[index]
			steering_angle = image_paths[index]  # this is ready to be used in the model