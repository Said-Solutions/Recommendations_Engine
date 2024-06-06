from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import pandas as pd

def train_content_based_model(products):
    # Concatenate 'category' column values for each product
    product_texts = products['category'].astype(str)
    
    # Vectorize product texts
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(product_texts)

    # Compute cosine similarity
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    return cosine_sim

def get_recommendations(product_id, cosine_sim, products):
    idx = products[products['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]
    product_indices = [i[0] for i in sim_scores]
    return products['product_id'].iloc[product_indices]

if __name__ == "__main__":
    # Read the large dataset files
    products = pd.read_csv('data/products_large.csv')
    cosine_sim = train_content_based_model(products)
