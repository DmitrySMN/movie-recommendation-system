import os
from dotenv import load_dotenv
from pinecone import Pinecone
from data_preprocessing import *


load_dotenv(dotenv_path='../../.env')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)

index = pc.Index("movies")

def upsert_tfidf_vectors_to_pinecone(data, reduced_tfidf_matrix, chunk_size=500):
    try:
        total_rows = reduced_tfidf_matrix.shape[0]
        upsert_data = []

        for i in range(total_rows):
            movie_id = str(data['id'].iloc[i])
            reduced_vector = reduced_tfidf_matrix[i].tolist()

            upsert_data.append({
                "id": movie_id,
                "values": reduced_vector,
                "metadata": { "movie_name": data['title'].iloc[i],
                             "movie_genre": data['genres'].iloc[i]}
            })

            if (i + 1) % chunk_size == 0 or (i + 1) == total_rows:
                index.upsert(vectors=upsert_data)
                upsert_data = []
                print(f"Upserted {i + 1}/{total_rows} vectors")

            print("Upsert completed successfully to Pinecone!")
    except Exception as e:
        print(str(e))

data = load_data_from_csv()

if data is None or data.empty:
    raise ValueError("Loaded data is empty or None.")

cosine_sim, tfidf_matrix = preprocess_data(data)

reduced_tfidf_matrix = reduce_dimensions(tfidf_matrix, n_components=1024)

upsert_tfidf_vectors_to_pinecone(data, reduced_tfidf_matrix)