import os
from uuid import uuid4
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

load_dotenv(dotenv_path='../../.env')

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies2048")
llama_embeddings = OllamaEmbeddings(model="llama3.2:1b")
vector_store = PineconeVectorStore(index, llama_embeddings)
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

def load_data_from_csv(file_path='../../dataset/movies.csv'):
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
    try:
        movie_info = data[data['title'] == movie_title].iloc[0]
        movie_id = movie_info['movieId']
        return movie_id
    except Exception as e:
        print(str(e))

def recommend_movies(movie_title, data, top_k):
    try:
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

            imdb_id = int(data[data['movieId'] == movie_id]['imdbId'].values[0])

            recommended_movies.append({
                'movie_id': movie_id,
                'movie_name': movie_name,
                'movie_genre': " ".join(movie_genre),
                'imdb_id': imdb_id
            })

        return recommended_movies
    except Exception as e:
        print(str(e))

def get_recommendations(movie_title):
    try:
        data = load_data_from_csv()
        recommended_movies = recommend_movies(movie_title, data, 5)
        return recommended_movies
    except Exception as e:
        print(str(e))
        return []

# def process_and_upload_to_pinecone(texts, ids):
#     embeddings = llama_embeddings.embed_documents(texts)
#     records = zip(ids, embeddings)
#     index.upsert(vectors=records)

def fill_index():
    try:
        df = load_data_from_csv()
        documents = []
        for i, row in df.iterrows():
            combined_features = row["combined_features"]
            movie_id = row['movieId']
            md = {
                "title": row["title"],
                "genres": row["genres"]
            }

            documents.append(Document(page_content=combined_features, metadata=md))
            print(f"document {row['movieId']} created")

        print("All documents created. Upsert into index started...")

        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        print("upsert into index successful")

    except Exception as e:
        print(str(e) + f" Movie id = {movie_id}")

def get_similar():
    results = vector_store.similarity_search("Drama")
    for res in results:
        print(f"* {res.page_content} [{res.metadata}]")
    print(results)

get_similar()
# fill_index()
