import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

from utils import data_batch_generator
from constants import DATA_DIR, INPUT_SHAPE, LEARNING_RATE, SAMPLES_PER_EPOCH, NB_EPOCH, BATCH_SIZE

def load_data():
    driving_df = pd.read_csv(DATA_DIR, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    X = driving_df[20000:22000][['center', 'left', 'right']]
    y = driving_df[20000:22000]['steering']

    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_validation, y_train, y_validation

def build_model():
    '''
    the model is based on the NVIDIA model

    the network has 9 layers, including a normalization layer, 5 convolutional
    layers and 3 fully connected layers

    the input image is an RGB with 160 by 320 pixels

    1. image normalization: the output is 3@160x320
    	values will have the same scale to avoid saturation and make gradeints work better

    2. convolution
    	designed to perform feature extraction
    	the layer configurations were choosen after a series of experiments
    	the first 3 layers were (5x5)kernels and strided (2x2) while the last 2 were 
    	(3x3) kernels and not strided(1x1)
    	to get the output size we use the formular [(W−K+2P)/S]+1, where W is the input volume, 
    	K is the kernel size, P is the padding  and S is the stride
    	the layers therefore lead to the following transformations
		    conv1: 5x5, filter: 24, strides: 2x2, activation: ELU input 3@66x200 => output 24@31x98
		    conv2: 5x5, filter: 36, strides: 2x2, activation: ELU input 24@31x98 => output 36@14x47
		    conv3: 5x5, filter: 48, strides: 2x2, activation: ELU input 36@14x47 => output 48@5x22
		    conv4: 3x3, filter: 64, strides: 1x1, activation: ELU input 48@5x22 => output 64@3x20
		    conv5: 3x3, filter: 64, strides: 1x1, activation: ELU input 64@3x20 => output 64@1x18

		the output of the last convolution layer is flattened to 1164 in order to work with fully connected layers
		the output will be 64 x 1 x 18 = 1152

	3. fully connected layers
		designed to act as the controller for steering
		the layers have decreasing number of neurons 100, 50, 10. they all use ELU activation function
    '''
    model = Sequential()

    # the normalization layer
    model.add(Lambda(lambda x: x/127.5-1.0, input_shape=INPUT_SHAPE))

    # the convolutional layers
    model.add(Conv2D(24, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(36, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(48, 5, 5, activation='elu', subsample=(2, 2)))
    model.add(Conv2D(64, 3, 3, activation='elu'))
    model.add(Conv2D(64, 3, 3, activation='elu'))


    model.add(Flatten())

    model.add(Dropout(0.2))  # dropout to reduce overfitting

    # fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.summary()

    return model

def train_model(model, X_train, X_valid, y_train, y_valid, steps_per_epoch=SAMPLES_PER_EPOCH):
	# adding a callback to save the model after every epoch
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

	# use mean squuared error to determine the accuracy of the model
	# use Adam optimizer to minimize the squared error
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))

	# fits a generator
	# this enables data preprosessing and training of the model to be done in parallel
	# in this case image augmentation will be done in the CPU while model training is in the GPU
	model.fit_generator(data_batch_generator(DATA_DIR, X_train, y_train, BATCH_SIZE, True),
                        steps_per_epoch,
                        NB_EPOCH,
                        max_q_size=1,
                        validation_data=data_batch_generator(DATA_DIR, X_valid, y_valid, BATCH_SIZE, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

def main():
    X_train, X_validation, y_train, y_validation = load_data()
    steps_per_epoch = len(X_train) / BATCH_SIZE
    model = build_model()
    train_model(model, X_train, X_validation, y_train, y_validation)

if __name__ == "__main__":
	main()