import numpy as np

from constants import IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS

def data_batch_generator(data_dir, image_paths, steering_angles, batch_size, is_training):
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