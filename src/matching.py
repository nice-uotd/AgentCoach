"""JD + 简历双匹配选题"""
import json
import logging
from pathlib import Path
from src.llm_client import chat_completion
from src.question_bank import get_question_bank, Question

logger = logging.getLogger(__name__)
DATA_DIR = Path(__file__).parent.parent / "data"

SYSTEM_PROMPT_RESUME_PARSE = """你是一个简历解析助手。请从简历中提取关键信息。
输出 JSON：
{
  "projects": ["项目名1", ...],
  "skills": ["技能1", ...],
  "keywords": ["技术关键词1", ...]
}
只输出 JSON。"""

SYSTEM_PROMPT_JD_PARSE = """你是一个 JD 解析助手。请从岗位描述中提取关键信息。
输出 JSON：
{
  "required_skills": ["必备技能1", ...],
  "bonus_skills": ["加分项1", ...],
  "focus_areas": ["考察重点1", ...]
}
只输出 JSON。"""


def parse_resume(resume_text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_RESUME_PARSE},
        {"role": "user", "content": resume_text}
    ]
    response = chat_completion(messages, temperature=0.3)
    return _safe_parse_json(response, {"projects": [], "skills": [], "keywords": []})


def parse_jd(jd_text: str) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_JD_PARSE},
        {"role": "user", "content": jd_text}
    ]
    response = chat_completion(messages, temperature=0.3)
    return _safe_parse_json(response, {"required_skills": [], "bonus_skills": [], "focus_areas": []})


def load_jd_templates() -> list[dict]:
    path = DATA_DIR / "jd_templates.json"
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def match_questions(resume_text: str, jd_text: str, top_k: int = 5) -> tuple[list[Question], dict]:
    """
    双匹配选题：简历 + JD → 个性化面试题
    返回 (选中的题目列表, 匹配元信息)
    """
    # 解析简历
    resume_info = parse_resume(resume_text)
    resume_keywords = (
        resume_info.get("keywords", []) +
        resume_info.get("skills", []) +
        resume_info.get("projects", [])
    )

    # 解析 JD
    jd_info = parse_jd(jd_text)
    jd_keywords = (
        jd_info.get("required_skills", []) +
        jd_info.get("bonus_skills", []) +
        jd_info.get("focus_areas", [])
    )

    # 构造融合 query：简历关键词 ∪ JD 关键词
    combined_query = " ".join(resume_keywords + jd_keywords)
    if not combined_query.strip():
        combined_query = resume_text[:200] + " " + jd_text[:200]

    # Hybrid 检索
    bank = get_question_bank()
    questions = bank.search(combined_query, top_k=top_k)

    meta = {
        "resume_keywords": resume_keywords[:15],
        "jd_keywords": jd_keywords[:10],
        "retrieval_mode": bank.retrieval_mode,
        "total_candidates": len(bank.questions),
    }

    return questions, meta


def _safe_parse_json(text: str, default: dict) -> dict:
    import re
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return default
