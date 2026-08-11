import os
import gc
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from src.ingestion.chunker import load_all_filings

COLLECTION_NAME = "sec_filings"
DB_PATH = "/tmp/qdrant_db"

def get_embedding_model():
    """Lightweight CPU embedding model."""
    print("[*] Loading local FastEmbed embedding model...")
    return FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5", threads=1)

def build_vector_store(chunks):
    """
    Saves vectors to /tmp/ disk on Linux/Cloud to keep RAM usage minimal (<150 MB).
    """
    os.makedirs(DB_PATH, exist_ok=True)
    embeddings = get_embedding_model()
    
    # Store directly in /tmp/ filesystem to save RAM
    client = QdrantClient(path=DB_PATH)
    
    if not client.collection_exists(COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )
        print(f"[*] Indexing {len(chunks)} chunks to disk at {DB_PATH}...")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        # Batch add to avoid memory spikes
        vector_store.add_documents(documents=chunks, batch_size=64)
    else:
        print("[*] Using existing vector store on disk...")
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=COLLECTION_NAME,
            embedding=embeddings
        )
        
    # Free memory
    gc.collect()
    print("[+] Successfully indexed all chunks into Qdrant!")
    return vector_store

if __name__ == "__main__":
    chunks = load_all_filings()
    db = build_vector_store(chunks)