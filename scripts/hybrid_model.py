import pandas as pd

def hybrid_recommendations(user_id, product_id, svd_model, cosine_sim, users, products):
    # Collaborative filtering scores
    collaborative_scores = [svd_model.predict(user_id, pid).est for pid in products['product_id']]
    
    # Content-based filtering scores
    content_scores = cosine_sim[products.index[products['product_id'] == product_id][0]]
    
    # Additional user features
    user_features = users[users['user_id'] == user_id]
    user_age = user_features['age'].values[0]
    user_gender = user_features['gender'].values[0]
    
    # Additional product features
    product_features = products[products['product_id'] == product_id]
    product_category = product_features['category'].values[0]
    product_price = product_features['price'].values[0]
    
    # Calculate hybrid scores
    hybrid_scores = {}
    for pid, collab_score, content_score in zip(products['product_id'], collaborative_scores, content_scores):
        # Additional user features
        user_factor = 1.0
        if user_age < 30:
            user_factor *= 1.1
        if user_gender == 'Female':
            user_factor *= 1.05
        
        # Additional product features
        product_factor = 1.0
        if product_category == 'Electronics':
            product_factor *= 1.2
        if product_price < 100:
            product_factor *= 1.1
        
        # Calculate hybrid score
        hybrid_score = (collab_score + content_score) / 2 * user_factor * product_factor
        hybrid_scores[pid] = hybrid_score
    
    # Sort hybrid scores
    sorted_hybrid_scores = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Return top recommendations
    return [pid for pid, score in sorted_hybrid_scores[:10]]
