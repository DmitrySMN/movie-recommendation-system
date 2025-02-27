import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone


load_dotenv(dotenv_path='../../.env')

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies")

def load_data_from_csv(file_path='../dataset/movies.csv'):
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} was not found.")

        data = pd.read_csv(filepath_or_buffer=file_path, low_memory=False)

        if data.empty:
            raise ValueError(f"The file {file_path} was loaded but it is empty.")

        return data
    except Exception as e:
        print(f"An error occurred while loading the data: {str(e)}")
        return None

def get_movie_id_by_title(movie_title, data):
    movie_info = data[data['title'] == movie_title].iloc[0]
    movie_id = movie_info['id']
    return movie_id

def recommend_movies(movie_title, data, top_k):
    movie_id = get_movie_id_by_title(movie_title, data)
    if not movie_id:
        return []

    query_response = index.query(
        id=str(movie_id),
        top_k=top_k + 1,
        include_metadata=True
    )

    if not query_response or 'matches' not in query_response:
        print("No matches found for the movie.")
        return []

    recommended_movies = []
    for match in query_response['matches'][1:top_k + 1]:
        metadata = match.get('metadata', {})
        movie_id = int(match['id'])
        movie_name = metadata.get('movie_name', 'Unknown Title')
        movie_genre = metadata.get('movie_genre', 'Unknown Genre').split()

        imdb_id = data[data['id'] == movie_id]['imdb_id'].values[0]

        recommended_movies.append({
            'movie_id': movie_id,
            'movie_name': movie_name,
            'movie_genre': " ".join(movie_genre),
            'imdb_id': imdb_id
        })

    return recommended_movies

def main():
    data = load_data_from_csv()
    movie_title = input("Enter movie title: ")

    recommended_movies = recommend_movies(movie_title, data, 12)
    print(recommended_movies)

if __name__ == '__main__':
    main()