import os
import pandas as pd
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_core.documents import Document

load_dotenv(dotenv_path='../../.env')

PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index("movies2048")
llama_embeddings = OllamaEmbeddings(model="llama3.2:1b")
vector_store = PineconeVectorStore(index, llama_embeddings)


class RecommendationSystem():
    @staticmethod
    def get_recommendation_by_message(query: str):
        docsearch = PineconeVectorStore.from_existing_index(
            index_name="movies2048",
            embedding=llama_embeddings,
            namespace="default"
        )
        retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
        retriever = docsearch.as_retriever()
        llm = ChatOllama(
            model='llama3.2:1b',
            temperature=0.0
        )
        combine_docs_chain = create_stuff_documents_chain(
            llm, retrieval_qa_chat_prompt
        )
        retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

        answer1_with_knowledge = retrieval_chain.invoke({"input": query})
        return answer1_with_knowledge
        
    @staticmethod
    def get_movie_id_by_title(movie_title, data):
        try:
            movie_info = data[data['title'] == movie_title].iloc[0]
            movie_id = movie_info['movieId']
            movie_combined_features = movie_info['combined_features']
            return movie_id, movie_combined_features
        except Exception as e:
            print(str(e))
            return None
        
    @staticmethod
    def get_similar(this_object, movie_title: str) -> list:
        data = RecommendationSystemUtils.load_data_from_csv()
        metadata = this_object.get_movie_id_by_title(movie_title, data)
        if metadata:
            results = vector_store.similarity_search(metadata[1], k=5)
            similar_movies = [i.metadata for i in results]
            return similar_movies
        else:
            return []
        

class RecommendationSystemUtils():
    @staticmethod
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