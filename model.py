import pandas as pd


def load_data():
	driving_df = pd.read_csv('data/driving_log.csv')

	print(driving_df.columns.tolist())

def main():
	load_data()

if __name__ == "__main__":
	main()