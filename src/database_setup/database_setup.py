from uuid import uuid4
from dotenv import load_dotenv
from pinecone import Pinecone
from data_preprocessing import *
from langchain_core.documents import Document
from langchain_pinecone import PineconeVectorStore
from langchain_ollama import OllamaEmbeddings, ChatOllama, OllamaLLM

load_dotenv(dotenv_path='../../.env')
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies2048")
llama_embeddings = OllamaEmbeddings(model="llama3.2:1b")
vector_store = PineconeVectorStore(index, llama_embeddings)


def get_documents():
    documents = []
    df = load_data_from_csv()
    df = df.dropna(subset=["combined_features"])

    for i, row in df.iterrows():
        combined_features = row["combined_features"]
        movie_id = row['movieId']
        md = {
            "title": row["title"],
            "genres": row["genres"]
        }
        documents.append(Document(page_content=combined_features, metadata=md))
        print(f"document {row['movieId']} created")

    return documents

def fill_index():
    try:
        documents = get_documents()

        print(f"All documents created. Document count = {len(documents)}. Upsert into index started...")

        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        print("upsert into index successful")

    except Exception as e:
        print(str(e))

def upsert_tfidf_vectors_to_pinecone(data, reduced_tfidf_matrix, chunk_size=250):
    try:
        total_rows = reduced_tfidf_matrix.shape[0]
        upsert_data = []

        for i in range(total_rows):
            movie_id = str(data['movieId'].iloc[i])
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

            print(f"Upsert completed successfully to Pinecone! id:{movie_id}")
    except Exception as e:
        print(str(e) + f"{movie_id}")

data = load_data_from_csv()

if data is None or data.empty:
    raise ValueError("Loaded data is empty or None.")

cosine_sim, tfidf_matrix = preprocess_data(data)

reduced_tfidf_matrix = reduce_dimensions(tfidf_matrix, n_components=2048)

upsert_tfidf_vectors_to_pinecone(data, reduced_tfidf_matrix)