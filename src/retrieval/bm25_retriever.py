"""BM25 关键词检索"""
import jieba
from rank_bm25 import BM25Okapi


def tokenize(text: str) -> list[str]:
    return list(jieba.cut_for_search(text.lower()))


class BM25Retriever:
    def __init__(self, documents: list[dict]):
        """
        documents: list of {"id": ..., "text": "用于索引的文本"}
        """
        self.doc_ids = [d["id"] for d in documents]
        corpus = [tokenize(d["text"]) for d in documents]
        self.bm25 = BM25Okapi(corpus)

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(doc_id, score), ...] 按分数降序"""
        tokens = tokenize(query)
        scores = self.bm25.get_scores(tokens)
        scored = sorted(zip(self.doc_ids, scores), key=lambda x: x[1], reverse=True)
        return scored[:top_k]
