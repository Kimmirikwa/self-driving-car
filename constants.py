# the directory of the data
DATA_DIR = 'log_data/driving_log.csv'

# the images are 66 by 200
IMAGE_HEIGHT = 66
IMAGE_WIDTH = 200

# RGB images
IMAGE_CHANNELS = 3

from keras.optimizers import Adam, RMSprop

SAMPLES_PER_EPOCH = 500  # the number steps in an epoch
NB_EPOCH = 100  # the number of iterations on the training data
BATCH_SIZE = 20

INPUT_SHAPE = (IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)

LEARNING_RATE = 1.0e-4

# the gradient descent learning rate
#LEARNING_RATE = [0.0001, 0.001, 0.003, 0.01, 0.03, 0.1] 
DROP0UT_RATE = [0.05, 0.1, 0.2, 0.3]
OPTIMIZERS = [Adam, RMSprop]
INIT = ['glorot_uniform', 'normal', 'uniform']

SHIFT_RANGE = [0.1, 0.15, 0.2, 0.25]

