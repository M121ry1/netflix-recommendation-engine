# netflix-recommendation-engine

Implementation of a Netflix movie recommendation algorithm using Matrix Factorization (SVD) and Cosine Similarity.

## Overview

This project implements a movie recommendation system inspired by Netflix's recommendation algorithms. It uses advanced machine learning techniques including Singular Value Decomposition (SVD) and Cosine Similarity to predict movie ratings and recommend content to users.

## Features

- **Matrix Factorization (SVD)**: Decomposes user-item rating matrices to uncover latent factors
- **Cosine Similarity**: Measures similarity between users or items for collaborative filtering
- **Rating Prediction**: Predicts unseen movie ratings based on user preferences
- **Personalized Recommendations**: Generates top-N movie recommendations for each user

## Technology Stack

- **Language**: Python
- **Key Libraries**: NumPy, Pandas, Scikit-learn, SciPy

## Project Structure

```
netflix-recommendation-engine/
├── README.md
├── requirements.txt
├── data/                    # Dataset directory
├── src/                     # Source code
│   ├── matrix_factorization.py
│   ├── similarity.py
│   ├── recommender.py

## Installation

Clone the repository:
```bash
git clone https://github.com/M121ry1/netflix-recommendation-engine.git
cd netflix-recommendation-engine
```

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Example

```python
from src.recommender import NetflixRecommender

# Initialize recommender
recommender = NetflixRecommender(n_factors=50)

# Load and train on data
recommender.fit(ratings_matrix)

# Get recommendations for a user
recommendations = recommender.recommend(user_id, n_recommendations=10)
print(recommendations)
```

## Algorithm Details

### Matrix Factorization (SVD)
Decomposes the user-item rating matrix into lower-dimensional latent factor matrices:
- Captures underlying patterns in user preferences
- Reduces dimensionality and improves computational efficiency
- Handles sparse rating data effectively

### Cosine Similarity
Calculates similarity between users or items:
- Measures angle between vectors in high-dimensional space
- Values range from -1 to 1 (or 0 to 1 for positive data)
- Used for content-based and collaborative filtering

## Performance Metrics

The model is evaluated using:
- **RMSE (Root Mean Squared Error)**: Measures prediction accuracy
- **Precision@K**: Percentage of recommended items that are relevant
- **Recall@K**: Percentage of relevant items in recommendations
- **NDCG (Normalized Discounted Cumulative Gain)**: Ranking quality

## Dataset

This project works with movie rating datasets (e.g., MovieLens, Netflix Prize Dataset format).

Expected format:
- User ID, Movie ID, Rating, Timestamp (optional)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.

## Author

M121ry1

## References

- Matrix Factorization: Koren, Y., Bell, R., & Volinsky, C. (2009)
- SVD: Golub, G. H., & Van Loan, C. F. (1989)
- Cosine Similarity in Information Retrieval: Tan, P. N., et al. (2005)
