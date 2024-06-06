from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
import joblib
import pandas as pd

def load_and_preprocess_data():
    # Load data from CSV files
    ratings = pd.read_csv('data/ratings_large.csv')
    users = pd.read_csv('data/users_large.csv')
    products = pd.read_csv('data/products_large.csv')

    # Additional preprocessing steps if necessary

    return ratings, users, products

def train_collaborative_filtering_model(train_data):
    # Load data into Surprise format
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(train_data[['user_id', 'product_id', 'rating']], reader)

    # Use SVD for collaborative filtering
    model = SVD()

    # Train model
    trainset = data.build_full_trainset()
    model.fit(trainset)

    return model

if __name__ == "__main__":
    ratings, users, products = load_and_preprocess_data()
    model = train_collaborative_filtering_model(ratings)
    joblib.dump(model, 'models/svd_model.pkl')
