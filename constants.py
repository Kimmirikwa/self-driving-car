# the directory of the data
DATA_DIR = 'data/driving_log.csv'

# the images are 66 by 200
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200

# RGB images
IMAGE_CHANNELS = 3

SAMPLES_PER_EPOCH = 50  # the number steps in an epoch
NB_EPOCH = 10  # the number of iterations on the training data
BATCH_SIZE = 10

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

# the gradient descent learning rate
LEARNING_RATE = 1.0e-4

SHIFT_RANGE = [0.1, 0.15, 0.2, 0.25]

