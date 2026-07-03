"""Hybrid 检索：BM25 + 向量 + RRF 融合"""
import logging
from src.config import RRF_K, BM25_WEIGHT, VECTOR_WEIGHT, VECTOR_MODEL
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.vector_retriever import VectorRetriever

logger = logging.getLogger(__name__)


def rrf_fusion(
    bm25_results: list[tuple[str, float]],
    vector_results: list[tuple[str, float]],
    k: int = 60,
    bm25_weight: float = 0.4,
    vector_weight: float = 0.6,
) -> list[tuple[str, float]]:
    """RRF 融合排序: score = Σ weight_i / (k + rank_i)"""
    scores: dict[str, float] = {}

    for rank, (doc_id, _) in enumerate(bm25_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + bm25_weight / (k + rank)

    for rank, (doc_id, _) in enumerate(vector_results, start=1):
        scores[doc_id] = scores.get(doc_id, 0) + vector_weight / (k + rank)

    sorted_results = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_results


class HybridRetriever:
    def __init__(self, documents: list[dict]):
        """
        documents: list of {"id": ..., "text": "检索用文本"}
        """
        self.bm25 = BM25Retriever(documents)
        self.vector = VectorRetriever(documents, model_name=VECTOR_MODEL)
        self.use_hybrid = self.vector.available

        if self.use_hybrid:
            logger.info("✅ Hybrid 检索就绪 (BM25 + 向量 + RRF)")
        else:
            logger.info("⚠️ 降级为纯 BM25 检索（向量模型不可用）")

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """返回 [(doc_id, rrf_score), ...] 融合排序后的结果"""
        bm25_results = self.bm25.search(query, top_k=top_k * 2)

        if not self.use_hybrid:
            return bm25_results[:top_k]

        vector_results = self.vector.search(query, top_k=top_k * 2)
        fused = rrf_fusion(
            bm25_results, vector_results,
            k=RRF_K, bm25_weight=BM25_WEIGHT, vector_weight=VECTOR_WEIGHT
        )
        return fused[:top_k]

    @property
    def mode(self) -> str:
        return "hybrid" if self.use_hybrid else "bm25_only"
