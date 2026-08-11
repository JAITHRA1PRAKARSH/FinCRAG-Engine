import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance
from src.ingestion.chunker import load_all_filings

COLLECTION_NAME = "sec_filings"

def get_embedding_model():
    """FastEmbed runs locally on CPU with zero API costs."""
    print("[*] Loading local FastEmbed embedding model...")
    return FastEmbedEmbeddings()

def build_vector_store(chunks):
    """
    Builds an in-memory Qdrant vector store from document chunks.
    Works seamlessly on Windows, Linux, and Cloud environments.
    """
    embeddings = get_embedding_model()
    client = QdrantClient(location=":memory:")
    
    # Create the in-memory collection
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    
    print(f"[*] Indexing {len(chunks)} chunks into Qdrant in-memory store...")
    vector_store = QdrantVectorStore(
        client=client,
        collection_name=COLLECTION_NAME,
        embedding=embeddings
    )
    
    vector_store.add_documents(documents=chunks)
    print("[+] Successfully indexed all chunks into Qdrant!")
    return vector_store

if __name__ == "__main__":
    chunks = load_all_filings()
    db = build_vector_store(chunks)
    
    query = "What are the primary supply chain risk factors and semiconductor risks?"
    results = db.similarity_search(query, k=2)
    for idx, doc in enumerate(results, 1):
        print(f"\n--- Result {idx} ---")
        print(f"Content: {doc.page_content[:250]}...")