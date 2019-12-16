import pandas as pd
from sklearn.model_selection import train_test_split


def load_data():
	# create a dataframe to hold the data
	# columns names are specified as shown below
	# the first 3 columns are for images from the 3 cameras and will acting as the features for our training model
	# 'steering' will be the labels, thus this is a supervised model training
	driving_df = pd.read_csv('data/driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

	X = driving_df[['center', 'left', 'right']]  # training features
	y = driving_df['steering']  # labels

	# split data to have training set and testing set
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

	return X_train, X_test, y_train, y_test

def main():
	# load and split data to training and testing set
	data = load_data()

if __name__ == "__main__":
	main()