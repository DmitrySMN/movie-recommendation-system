import logging
import os
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO)

def load_data_from_csv(file_path='../../dataset/movies.csv'):
    try:
        logging.info(f"Data Loading Started from {file_path}")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        data = pd.read_csv(filepath_or_buffer=file_path, low_memory=False)

        if data.empty:
            raise ValueError(f"The file {file_path} was loaded but it is empty.")

        logging.info(f"Data Loaded Successfully! Shape of the dataset: {data.shape}")
        return data
    except Exception as e:
        logging.exception(f"An error occurred while loading the data: {str(e)}")
        return None


def preprocess_data(data, max_features=7000, ngram_range=(1, 2)):
    try:
        logging.info("Data preprocessing started")
        if data.empty:
            raise ValueError("The dataset is empty.")

        logging.info(f"Input data shape: {data.shape}")

        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features, ngram_range=ngram_range)

        logging.info("Fitting TF-IDF Vectorizer...")

        tfidf_matrix = vectorizer.fit_transform(data['overview'])

        logging.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

        logging.info(f"cosine similarity matrix shape: {cosine_sim.shape}")

        logging.info("Data preprocessing completed successfully")
        return cosine_sim, tfidf_matrix
    except Exception as e:
        logging.error(f"Error occurred during preprocessing: {str(e)}")
        raise e

def reduce_dimensions(tfidf_matrix, n_components):
    print("Starting dimensionality reduction using TruncatedSVD...")
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    reduced_tfidf_matrix = svd.fit_transform(tfidf_matrix)
    print(f"Dimensionality reduced from {tfidf_matrix.shape[1]} to {n_components} components.")
    return reduced_tfidf_matrix