"""评估报告 + 6维能力画像"""
import json
from src.config import SCORE_DIMENSIONS
from src.interview_agent import InterviewSession
from src.llm_client import chat_completion

SYSTEM_PROMPT_EVALUATION = """你是一位面试评估专家。请根据面试记录生成结构化能力评估报告。

## 评估维度（6维）
- 技术深度 (20%): 对技术原理的理解、实践深度
- 工程经验 (20%): 生产级工程实践、部署运维经验
- 表达清晰度 (15%): 逻辑是否清晰、是否结构化
- STAR 完整性 (15%): 是否有情境、任务、行动、结果
- 系统设计能力 (15%): 架构思维、可扩展性考虑
- RAG/Agent 专项 (15%): RAG 检索/Agent 编排方向的专项能力

## 面试记录
{interview_records}

## 参考答案对比
{answer_comparison}

## 输出 JSON
{{
  "overall_score": 总分(1-10),
  "dimensions": {{
    "technical_depth": {{"score": 分数, "comment": "评语"}},
    "engineering_exp": {{"score": 分数, "comment": "评语"}},
    "clarity": {{"score": 分数, "comment": "评语"}},
    "star_completeness": {{"score": 分数, "comment": "评语"}},
    "system_design": {{"score": 分数, "comment": "评语"}},
    "rag_agent_skill": {{"score": 分数, "comment": "评语"}}
  }},
  "strengths": ["优点1", "优点2", ...],
  "weaknesses": ["薄弱点1", "薄弱点2", ...],
  "missed_points": ["遗漏知识点1", "遗漏知识点2", ...],
  "study_path": ["推荐复习路径1", "路径2", ...],
  "next_round_suggestions": ["下轮建议1", "建议2", ...]
}}
只输出 JSON。"""


def generate_evaluation(session: InterviewSession) -> dict:
    records_text = ""
    answer_comparison = ""

    for i, record in enumerate(session.records):
        if not record.exchanges:
            continue
        records_text += f"\n## 题目 {i+1}: {record.question.title}\n"
        records_text += f"类别: {record.question.category} | 难度: {record.question.difficulty}\n"
        for ex in record.exchanges:
            role_name = "面试官" if ex["role"] == "interviewer" else "候选人"
            records_text += f"[{role_name}]: {ex['content']}\n"
        if record.score:
            records_text += f"实时评分: {record.score}/10\n"
        records_text += "---\n"

        answer_comparison += f"\n### 题目 {i+1}: {record.question.title}\n"
        answer_comparison += f"参考答案: {record.question.answer_key_points}\n"
        answer_comparison += f"评估标准: {'; '.join(record.question.rubric)}\n---\n"

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT_EVALUATION.format(
            interview_records=records_text,
            answer_comparison=answer_comparison
        )},
        {"role": "user", "content": "请生成6维能力评估报告。"}
    ]

    response = chat_completion(messages, temperature=0.3, max_tokens=2048)
    result = _safe_parse(response)
    return result


def compute_weighted_score(dimensions: dict) -> float:
    total = 0.0
    for key, config in SCORE_DIMENSIONS.items():
        dim_data = dimensions.get(key, {})
        score = dim_data.get("score", 0)
        total += score * config["weight"]
    return round(total, 1)


def _safe_parse(text: str) -> dict:
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
    return {
        "overall_score": 0,
        "dimensions": {k: {"score": 0, "comment": "评估失败"} for k in SCORE_DIMENSIONS},
        "strengths": [],
        "weaknesses": ["评估报告生成失败，请重试"],
        "missed_points": [],
        "study_path": [],
        "next_round_suggestions": []
    }
