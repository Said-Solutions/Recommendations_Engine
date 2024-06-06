import os
import pandas as pd
import numpy as np

# Generate user IDs, product IDs, and ratings
num_users = 1000
num_products = 500
num_interactions = 10000

user_ids = np.random.randint(1, num_users + 1, size=num_interactions)
product_ids = np.random.randint(1, num_products + 1, size=num_interactions)
ratings = np.random.randint(1, 6, size=num_interactions)  # Ratings from 1 to 5

# Create interactions dataframe
interactions_df = pd.DataFrame({
    'user_id': user_ids,
    'product_id': product_ids,
    'rating': ratings
})

# Generate additional user features (e.g., age, gender)
user_features = {
    'user_id': np.arange(1, num_users + 1),
    'age': np.random.randint(18, 70, size=num_users),
    'gender': np.random.choice(['Male', 'Female'], size=num_users)
}
user_features_df = pd.DataFrame(user_features)

# Generate additional product features (e.g., category, price)
product_features = {
    'product_id': np.arange(1, num_products + 1),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home', 'Beauty'], size=num_products),
    'price': np.random.uniform(10, 500, size=num_products)
}
product_features_df = pd.DataFrame(product_features)

# Ensure that the data folder exists
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Save data to CSV files in the data folder
interactions_df.to_csv(os.path.join(data_folder, 'ratings_large.csv'), index=False)
user_features_df.to_csv(os.path.join(data_folder, 'users_large.csv'), index=False)
product_features_df.to_csv(os.path.join(data_folder, 'products_large.csv'), index=False)

print("Large datasets generated successfully!")
