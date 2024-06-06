import pandas as pd
from surprise import Reader, Dataset
from surprise.model_selection import train_test_split
from surprise import accuracy
from surprise import SVD  

# Load data from CSV into a DataFrame
data = pd.read_csv('data/ratings_large.csv')  

# Print the first few rows of the DataFrame to verify data loading
print(data.head())

# Define the Reader object
reader = Reader(rating_scale=(1, 5))

# Load the data into a Surprise dataset
surprise_data = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)

# Split the data into train and test sets
trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)

# Example: Train an SVD model on the trainset
model = SVD()
model.fit(trainset)

# Make predictions on the test set
predictions = model.test(testset)

# Evaluate the model using RMSE
rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')
