import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D, Flatten, Dense
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

from utils import data_batch_generator
from constants import DATA_DIR, INPUT_SHAPE, LEARNING_RATE, SAMPLES_PER_EPOCH, NB_EPOCH, BATCH_SIZE

def load_data():
	# create a dataframe to hold the data
	# columns names are specified as shown below
	# the first 3 columns are for paths of images from the 3 cameras and will acting as the features for our training model
	# 'steering' will be the labels, thus this is a supervised model training
	driving_df = pd.read_csv(DATA_DIR, names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

	X = driving_df[['center', 'left', 'right']]  # training features
	y = driving_df['steering']  # labels

	# split data to have training set and validation set
	X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.3, random_state=42)

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
		    conv1: 5x5, filter: 24, strides: 2x2, activation: ELU input 3@160x320 => output 24@78x158
		    conv2: 5x5, filter: 36, strides: 2x2, activation: ELU input 24@78x158 => output 36@37x77
		    conv3: 5x5, filter: 48, strides: 2x2, activation: ELU input 36@37x77 => output 48@17x37
		    conv4: 3x3, filter: 64, strides: 1x1, activation: ELU input 48@17x37 => output 64@15x35
		    conv5: 3x3, filter: 64, strides: 1x1, activation: ELU input 64@15x35 => output 64@13x33

		the output of the last convolution layer is flattened to 1164 in order to work with fully connected layers
		the output will be 64 x 13 x 33 = 27456

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

    # fully connected layers
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))

    model.summary()

    return model

def train_model(model, X_train, X_valid, y_train, y_valid):
	# adding a callback to save the model after every epoch
	checkpoint = ModelCheckpoint('model-{epoch:03d}.h5',
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             mode='auto')

	# use mean squuared error to determine the accuracy of the model
	# use Adam optimizer to minimize the squared error
	model.compile(loss='mean_squared_error', optimizer=Adam(lr=LEARNING_RATE))

	model.fit_generator(data_batch_generator(DATA_DIR, X_train, y_train, BATCH_SIZE, True),
                        SAMPLES_PER_EPOCH,
                        NB_EPOCH,
                        max_q_size=1,
                        validation_data=data_batch_generator(DATA_DIR, X_valid, y_valid, BATCH_SIZE, False),
                        nb_val_samples=len(X_valid),
                        callbacks=[checkpoint],
                        verbose=1)

def main():
	# load and split data to training and testing set
	data = load_data()

	# building the model
	model = build_model()

	# train the model using data and save it
	#train_model(model, *data)

if __name__ == "__main__":
	main()