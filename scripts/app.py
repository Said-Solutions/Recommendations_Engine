import joblib
import pandas as pd
from flask import Flask, request, jsonify
from content_based_filtering import train_content_based_model
from hybrid_model import hybrid_recommendations

app = Flask(__name__)

# Load necessary data and models
svd_model = joblib.load('models/svd_model.pkl')
products = pd.read_csv('data/products_large.csv')
users = pd.read_csv('data/users_large.csv')
cosine_sim = train_content_based_model(products)

@app.route('/recommend', methods=['GET'])
def recommend():
    # check for user_id 
    user_id = 1
    
    # Check for product_id 
    product_id = 1
       
    # Generate recommendations
    recommendations = hybrid_recommendations(user_id, product_id, svd_model, cosine_sim, users, products)
    
    # Return recommendations
    return jsonify({'recommendations': recommendations})

if __name__ == "__main__":
    app.run(debug=True)
