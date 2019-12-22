import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Lambda, Conv2D

from constants import INPUT_SHAPE


def load_data():
	# create a dataframe to hold the data
	# columns names are specified as shown below
	# the first 3 columns are for paths of images from the 3 cameras and will acting as the features for our training model
	# 'steering' will be the labels, thus this is a supervised model training
	driving_df = pd.read_csv('data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

	X = driving_df[['center', 'left', 'right']][:200]  # training features
	y = driving_df['steering'][:200]  # labels

	# split data to have training set and testing set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	return X_train, X_test, y_train, y_test

def build_model():
    '''
    the model is based on the NVIDIA model

    the network has 9 layers, including a normalization layer, 5 convolutional
    layers and 3 fully connected layers

    the input image is an RGB with 66 by 200 pixels

    1. image normalization: the output is 3@66x200
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

    return model

def main():
	# load and split data to training and testing set
	data = load_data()

	# building the model
	model = build_model()

if __name__ == "__main__":
	main()