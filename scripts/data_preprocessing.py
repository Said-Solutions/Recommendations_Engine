import pandas as pd
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split

def load_and_preprocess_data():
    # Load datasets
    ratings = pd.read_csv('data/ratings_large.csv')
    products = pd.read_csv('data/products_large.csv')
    users = pd.read_csv('data/users_large.csv')  # Add loading users_large.csv

    # Data cleaning
    ratings.dropna(inplace=True)
    products.dropna(inplace=True)
    users.dropna(inplace=True)  # Add dropping NA values for users

    # Merge datasets
    data = pd.merge(ratings, products, on='product_id')
    data = pd.merge(data, users, on='user_id')  # Merge with users dataset

    # Create Surprise dataset
    reader = Reader(rating_scale=(1, 5))
    surprise_data = Dataset.load_from_df(data[['user_id', 'product_id', 'rating']], reader)

    # Train-test split
    trainset, testset = train_test_split(surprise_data, test_size=0.2, random_state=42)

    return trainset, testset

if __name__ == "__main__":
    # Load and preprocess data
    trainset, _ = load_and_preprocess_data()
    print("Data loaded and preprocessed successfully.")
