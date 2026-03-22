import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from rag.embeddings import embed


class HybridVectorStore:
    """FAISS semantic search + BM25 keyword search, fused via Reciprocal Rank Fusion."""

    def __init__(self):
        self._reset()

    def _reset(self):
        self._docs:       list = []
        self._index:      faiss.Index = None
        self._bm25:       BM25Okapi   = None
        self._tokenized:  list = []
        self._source_label: str = ""

    def clear(self):
        """Wipe the entire knowledge base so a new source can be loaded."""
        self._reset()

    @property
    def doc_count(self) -> int:
        return len(set(d.get("source", "") for d in self._docs)) if self._docs else 0

    @property
    def chunk_count(self) -> int:
        return len(self._docs)

    @property
    def source_label(self) -> str:
        return self._source_label

    def add_documents(self, chunks: list, source_label: str = ""):
        self._docs.extend(chunks)
        self._source_label = source_label or (chunks[0].get("source", "") if chunks else "")
        texts        = [c["page_content"] for c in self._docs]
        vectors      = embed(texts)
        dim          = vectors.shape[1]
        self._index  = faiss.IndexFlatIP(dim)
        self._index.add(vectors)
        self._tokenized = [t.lower().split() for t in texts]
        self._bm25      = BM25Okapi(self._tokenized)

    def hybrid_search(self, query: str, k: int = 5) -> list:
        if not self._docs:
            return []
        k = min(k, len(self._docs))

        q_vec            = embed([query])
        scores, idxs     = self._index.search(q_vec, min(k * 2, len(self._docs)))
        sem_ranks        = {int(idxs[0][r]): r for r in range(len(idxs[0]))}

        bm25_scores      = self._bm25.get_scores(query.lower().split())
        bm25_order       = np.argsort(bm25_scores)[::-1][:k * 2]
        bm25_ranks       = {int(bm25_order[r]): r for r in range(len(bm25_order))}

        rrf_k    = 60
        all_ids  = set(sem_ranks) | set(bm25_ranks)
        rrf      = {i: 1/(rrf_k + sem_ranks.get(i,999)) + 1/(rrf_k + bm25_ranks.get(i,999)) for i in all_ids}
        top_ids  = sorted(rrf, key=lambda i: rrf[i], reverse=True)[:k]

        results = []
        for idx in top_ids:
            doc = dict(self._docs[idx])
            doc["score"] = round(rrf[idx], 4)
            results.append(doc)
        return results
