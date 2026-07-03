"""向量语义检索（sentence-transformers）"""
import logging
import numpy as np

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False
    logger.warning("sentence-transformers 不可用，向量检索将降级为纯 BM25")


class VectorRetriever:
    def __init__(self, documents: list[dict], model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        """
        documents: list of {"id": ..., "text": "用于编码的文本"}
        """
        self.doc_ids = [d["id"] for d in documents]
        self.available = _ST_AVAILABLE

        if not self.available:
            self.embeddings = None
            return

        try:
            logger.info(f"加载向量模型: {model_name}")
            self.model = SentenceTransformer(model_name)
            texts = [d["text"] for d in documents]
            self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            logger.info(f"题库向量编码完成: {len(texts)} 条, 维度 {self.embeddings.shape[1]}")
        except Exception as e:
            logger.warning(f"向量模型加载失败: {e}，降级为纯 BM25")
            self.available = False
            self.embeddings = None

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """返回 [(doc_id, score), ...] 按相似度降序"""
        if not self.available:
            return []

        query_emb = self.model.encode([query], normalize_embeddings=True)
        similarities = np.dot(self.embeddings, query_emb.T).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append((self.doc_ids[idx], float(similarities[idx])))
        return results
