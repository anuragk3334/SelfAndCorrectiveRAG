from langchain_pinecone import PineconeVectorStore
import os
from dotenv  import load_dotenv
from pinecone import Pinecone
from core.embeddings import embeddings

load_dotenv()
pc=Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

def write_embeddings_to_pinecone(chunks):
    """Writes document chunks and their embeddings to Pinecone vector store.

    Args:
        chunks (List): List of document chunks.
        embeddings (List): List of embeddings corresponding to the document chunks.
    """
    PineconeVectorStore.from_documents(
        documents=chunks,embedding=embeddings,
        index_name="diabetes")
    
def query_pinecone(query, k=5):
    """Queries the Pinecone vector store for relevant documents.

    Args:
        query (str): The query string to search for relevant documents.
        k (int): Number of documents to retrieve."""
    vectorStore= PineconeVectorStore(embedding=embeddings,index_name="diabetes")
    docs=vectorStore.similarity_search(query, k=k)
    return docs
        
    
   
    