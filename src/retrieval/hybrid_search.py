from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from src.ingestion.chunker import load_all_filings
from src.retrieval.vector_store import build_vector_store

class HybridRetriever:
    def __init__(self, chunks):
        print("[*] Initializing Hybrid Retriever...")
        self.chunks = chunks
        
        # 1. Build Vector Store
        self.vector_store = build_vector_store(self.chunks)
        
        # 2. Build BM25 Keyword Index
        print("[*] Building BM25 exact-keyword index...")
        tokenized_corpus = [doc.page_content.lower().split() for doc in self.chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        # 3. Load Cross-Encoder Re-Ranker
        print("[*] Loading Cross-Encoder Re-ranking model (MiniLM)...")
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
        
        print("[+] Retriever Engine Fully Armed!")

    def rrf_merge(self, vector_results, bm25_results, k=60, top_n=15):
        """Merges vector search and BM25 search using Reciprocal Rank Fusion."""
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

    def search(self, query: str, final_k: int = 6):
        """
        1. Wide Net: Retrieve top 15 from Vector and BM25.
        2. Merge: Reciprocal Rank Fusion (RRF).
        3. Re-Rank: MiniLM Cross-Encoder.
        """
        print(f"\n[*] Executing Hybrid Search for: '{query}'")
        
        # 1. Fetch Candidates
        vector_docs = self.vector_store.similarity_search(query, k=15)
        bm25_docs = self.bm25.get_top_n(query.lower().split(), self.chunks, n=15)
        
        # 2. RRF Merge
        fused_docs = self.rrf_merge(vector_docs, bm25_docs, top_n=15)
        
        # 3. Cross-Encoder Re-Ranking
        print("[*] Re-Ranking candidates with Cross-Encoder...")
        pairs = [[query, doc.page_content] for doc in fused_docs]
        scores = self.reranker.predict(pairs)
        
        scored_docs = list(zip(fused_docs, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        return [doc for doc, score in scored_docs[:final_k]]