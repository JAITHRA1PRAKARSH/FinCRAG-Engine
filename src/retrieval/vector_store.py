import os
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_qdrant import Qdrant
from src.ingestion.chunker import load_all_filings, create_sec_chunks

COLLECTION_NAME = "sec_filings"
DB_PATH = "data/qdrant_db"

def get_embedding_model():
    """
    Uses FastEmbed (runs locally on CPU, zero API cost, high accuracy).
    Default model: BAAI/bge-small-en-v1.5
    """
    print("[*] Loading local FastEmbed embedding model...")
    return FastEmbedEmbeddings()

def build_vector_store(chunks, force_recreate: bool = False):
    """
    Embeds chunks and stores them persistently in an on-disk local Qdrant database.
    """
    os.makedirs(DB_PATH, exist_ok=True)
    embeddings = get_embedding_model()
    
    print(f"[*] Storing {len(chunks)} chunks into Qdrant collection '{COLLECTION_NAME}' at '{DB_PATH}'...")
    print("    (Please wait ~30-45 seconds for CPU embedding to finish...)")
    
    qdrant = Qdrant.from_documents(
        documents=chunks,
        embedding=embeddings,
        path=DB_PATH,
        collection_name=COLLECTION_NAME,
        force_recreate=force_recreate
    )
    print("[+] Successfully embedded and saved all chunks to local disk!")
    return qdrant

def get_vector_store():
    """
    Loads the existing Qdrant vector store from disk without re-embedding.
    """
    embeddings = get_embedding_model()
    return Qdrant.from_existing_collection(
        embedding=embeddings,
        path=DB_PATH,
        collection_name=COLLECTION_NAME
    )

if __name__ == "__main__":
    # 1. Load ALL company filings (AAPL + MSFT)
    chunks = load_all_filings()
    
    # 2. Build & Save Database with both companies
    db = build_vector_store(chunks, force_recreate=True)
    
    # 3. Test a sample query against the database
    print("\n" + "="*60)
    print("TESTING VECTOR SEARCH FROM LOCAL QDRANT DB:")
    print("="*60)
    query = "What are the primary supply chain risk factors and semiconductor risks?"
    print(f"Query: '{query}'\n")
    
    results = db.similarity_search_with_score(query, k=2)
    for idx, (doc, score) in enumerate(results, 1):
        print(f"--- Result {idx} (Similarity Score: {score:.4f}) ---")
        print(f"Metadata: {doc.metadata}")
        print(f"Content snippet: {doc.page_content[:250]}...\n")
    print("="*60)