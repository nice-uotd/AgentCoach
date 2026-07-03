"""题库加载 + Hybrid 检索"""
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from src.retrieval.hybrid import HybridRetriever

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"


@dataclass
class Question:
    id: str
    title: str
    category: str
    difficulty: str
    tags: list[str] = field(default_factory=list)
    rubric: list[str] = field(default_factory=list)
    answer_key_points: str = ""
    followups: list[str] = field(default_factory=list)


class QuestionBank:
    def __init__(self):
        self.questions: list[Question] = []
        self._id_map: dict[str, Question] = {}
        self._retriever: HybridRetriever | None = None
        self._load()

    def _load(self):
        path = DATA_DIR / "questions.json"
        if not path.exists():
            logger.error(f"题库文件不存在: {path}")
            return

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            q = Question(
                id=item["id"],
                title=item["title"],
                category=item["category"],
                difficulty=item.get("difficulty", "medium"),
                tags=item.get("tags", []),
                rubric=item.get("rubric", []),
                answer_key_points=item.get("answer_key_points", ""),
                followups=item.get("followups", []),
            )
            self.questions.append(q)
            self._id_map[q.id] = q

        # 构建检索索引
        documents = []
        for q in self.questions:
            text = f"{q.title} {q.category} {' '.join(q.tags)} {' '.join(q.rubric)}"
            documents.append({"id": q.id, "text": text})

        self._retriever = HybridRetriever(documents)
        logger.info(f"题库加载完成: {len(self.questions)} 道题, 检索模式: {self._retriever.mode}")

    def search(self, query: str, top_k: int = 10, category_filter: str = "") -> list[Question]:
        """Hybrid 检索题目"""
        if not self._retriever:
            return self.questions[:top_k]

        results = self._retriever.search(query, top_k=top_k * 2)
        matched = []
        for doc_id, score in results:
            q = self._id_map.get(doc_id)
            if q is None:
                continue
            if category_filter and q.category != category_filter:
                continue
            matched.append(q)
            if len(matched) >= top_k:
                break
        return matched

    def get_by_id(self, qid: str) -> Question | None:
        return self._id_map.get(qid)

    def get_by_category(self, category: str) -> list[Question]:
        return [q for q in self.questions if q.category == category]

    @property
    def retrieval_mode(self) -> str:
        return self._retriever.mode if self._retriever else "none"


_bank: QuestionBank | None = None


def get_question_bank() -> QuestionBank:
    global _bank
    if _bank is None:
        _bank = QuestionBank()
    return _bank
