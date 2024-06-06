# AI-powered Recommendation Engine

## Project Structure


## Setup Instructions

1. Generate data:
    ```bash
    python generate_large_data.py
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Preprocess data:
    ```bash
    python scripts/data_preprocessing.py
    ```

4. Train models:
    ```bash
    python scripts/collaborative_filtering.py
    python scripts/content_based_filtering.py
    ```

5. Get hybrid recommendations:
    ```bash
    python scripts/hybrid_model.py
    ```

6. Run the Flask app:
    ```bash
    python app.py
    ```

7. Evaluate the model:
    ```bash
    python scripts/evaluate.py
    ```

## Usage

Access the recommendation engine at `http://127.0.0.1:5000/recommend?user_id=1`.

