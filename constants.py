# the directory of the data
DATA_DIR = 'data/driving_log.csv'

# the images are 66 by 200
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200

# RGB images
IMAGE_CHANNELS = 3

SAMPLES_PER_EPOCH = 200
NB_EPOCH = 10  # the number of epochs
BATCH_SIZE = 30

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# the gradient descent learning rate
LEARNING_RATE = 1.0e-4

