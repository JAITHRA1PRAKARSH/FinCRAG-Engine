from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.ingestion.chunker import load_all_filings, create_sec_chunks
from src.retrieval.vector_store import get_vector_store

class HybridRetriever:
    def __init__(self, chunks):
        print("[*] Initializing Hybrid Retriever...")
        self.chunks = chunks
        self.vector_store = get_vector_store()
        
        # 1. Build BM25 Keyword Index
        print("[*] Building BM25 exact-keyword index...")
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 2. Load Cross-Encoder Re-Ranker (runs locally)
        print("[*] Loading Cross-Encoder Re-ranking model (MiniLM)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print("[+] Retriever Engine Fully Armed!")

    def rrf_merge(self, vector_results, bm25_results, k=60, top_n=15):
        """Merges results using Reciprocal Rank Fusion."""
        rrf_scores = {}
        
        for rank, doc in enumerate(vector_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, {"doc": doc, "score": 0.0})
            rrf_scores[doc_id]["score"] += 1.0 / (k + rank + 1)
            
        for rank, doc in enumerate(bm25_results):
            doc_id = id(doc)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, {"doc": doc, "score": 0.0})
            rrf_scores[doc_id]["score"] += 1.0 / (k + rank + 1)
            
        sorted_docs = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)
        return [item["doc"] for item in sorted_docs[:top_n]]

    def search(self, query: str, final_k: int = 3):
        """
        1. Wide Net: Pull top 15 from Vector and top 15 from BM25.
        2. Merge: Combine via RRF.
        3. Re-Rank: Use Cross-Encoder to find the absolute top best.
        """
        print(f"\n[*] Executing Hybrid Search for: '{query}'")
        
        # 1. Fetch from Vector & BM25
        vector_docs = [doc for doc, _ in self.vector_store.similarity_search_with_score(query, k=15)]
        bm25_docs = self.bm25.get_top_n(query.lower().split(), self.chunks, n=15)
        
        # 2. RRF Merge
        fused_docs = self.rrf_merge(vector_docs, bm25_docs, top_n=15)
        
        # 3. Cross-Encoder Re-Ranking
        print("[*] Re-Ranking the top 15 chunks with Cross-Encoder...")
        pairs = [[query, doc.page_content] for doc in fused_docs]
        scores = self.reranker.predict(pairs)
        
        # Attach scores to documents and sort descending
        scored_docs = list(zip(fused_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Return only the top 'final_k' chunks
        return [doc for doc, score in scored_docs[:final_k]]

if __name__ == "__main__":
    # Test our Hybrid Search engine with all companies loaded
    chunks = load_all_filings()
    
    retriever = HybridRetriever(chunks)
    
    print("\n" + "="*70)
    print("TESTING 2-STAGE RETRIEVAL (Hybrid + Cross-Encoder):")
    print("="*70)
    
    test_query = "What are the risks related to tariffs, international trade, and China?"
    
    top_docs = retriever.search(test_query, final_k=2)
    for idx, doc in enumerate(top_docs, 1):
        print(f"\n--- Final Re-Ranked Result {idx} ---")
        print(f"Content snippet: {doc.page_content[:400]}...")
    print("\n" + "="*70)