from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

from .documents import DocumentChunk


@dataclass
class SearchHit:
    chunk: DocumentChunk
    score: float


class Retriever:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=1)
        corpus = [chunk.text for chunk in chunks] or [""]
        self.matrix = self.vectorizer.fit_transform(corpus)
        self.mode = "tfidf"
        self.index = None
        self.svd = None
        if len(chunks) >= 4:
            try:
                import faiss

                dims = min(256, max(2, min(self.matrix.shape) - 1))
                self.svd = TruncatedSVD(n_components=dims, random_state=42)
                dense = self.svd.fit_transform(self.matrix)
                dense = normalize(dense).astype("float32")
                self.index = faiss.IndexFlatIP(dense.shape[1])
                self.index.add(dense)
                self.mode = "faiss"
            except Exception:
                self.index = None
                self.svd = None

    def search(self, query: str, k: int = 6) -> list[SearchHit]:
        if not self.chunks:
            return []
        q = self.vectorizer.transform([query])
        if self.index is not None and self.svd is not None:
            dense = normalize(self.svd.transform(q)).astype("float32")
            scores, indices = self.index.search(dense, min(k, len(self.chunks)))
            return [
                SearchHit(self.chunks[int(idx)], float(score))
                for score, idx in zip(scores[0], indices[0])
                if idx >= 0
            ]
        sims = cosine_similarity(q, self.matrix)[0]
        order = np.argsort(sims)[::-1][:k]
        return [SearchHit(self.chunks[int(idx)], float(sims[int(idx)])) for idx in order]
