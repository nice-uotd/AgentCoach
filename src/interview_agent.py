"""面试 Agent —— ReAct 策略驱动的面试官"""
import json
from dataclasses import dataclass, field
from src.config import MAX_QUESTIONS, MAX_FOLLOWUPS
from src.question_bank import Question
from src.llm_client import chat_completion


@dataclass
class QuestionRecord:
    question: Question
    exchanges: list[dict] = field(default_factory=list)
    followup_count: int = 0
    score: int | None = None
    feedback: str = ""


@dataclass
class InterviewSession:
    resume_text: str = ""
    jd_text: str = ""
    resume_keywords: list[str] = field(default_factory=list)
    jd_keywords: list[str] = field(default_factory=list)
    selected_questions: list[Question] = field(default_factory=list)
    records: list[QuestionRecord] = field(default_factory=list)
    current_question_idx: int = 0
    current_followup_count: int = 0
    is_finished: bool = False
    chat_history: list[dict] = field(default_factory=list)
    retrieval_mode: str = ""

    @property
    def current_record(self) -> QuestionRecord | None:
        if 0 <= self.current_question_idx < len(self.records):
            return self.records[self.current_question_idx]
        return None

    @property
    def progress_text(self) -> str:
        total = len(self.selected_questions)
        current = min(self.current_question_idx + 1, total)
        return f"第 {current}/{total} 题 | 追问 {self.current_followup_count}/{MAX_FOLLOWUPS}"


SYSTEM_PROMPT_INTERVIEWER = """你是一位资深的 AI 技术面试官，正在面试一位候选人。

## 岗位信息
{jd_summary}

## 面试规则
1. 根据候选人回答质量 + 参考答案对比决定下一步动作
2. 评判标准：
   - 覆盖参考答案 ≥60% 核心要点 + 有数据/案例 → 高分(7-10)，换下一题
   - 正确但笼统，只覆盖 1-2 个要点 → 中等(4-6)，针对缺失要点追问
   - 偏题/空泛 → 低分(1-3)，给提示并追问

## 追问策略
- 回答优秀 → 简短肯定 + 换题
- 回答中等 → 针对未提及的核心要点追问
- 回答较差 → "能用 STAR 法则组织吗？" + 方向性提示

## 个性化追问（基于简历）
根据候选人简历把问题和项目经历关联。

## 当前面试题
{current_question}

## 参考答案核心要点（不要告诉候选人）
{answer_key_points}

## 候选人简历
{resume_summary}

## 输出 JSON
{{
  "action": "ask_new" | "follow_up" | "end_topic",
  "message": "面试官发言",
  "score": 本次回答得分(1-10) 或 null,
  "feedback": "内部评价" 或 null
}}
只输出 JSON。"""


def start_interview(resume_text: str, jd_text: str, questions: list[Question],
                    resume_keywords: list[str], jd_keywords: list[str],
                    retrieval_mode: str = "") -> InterviewSession:
    session = InterviewSession(
        resume_text=resume_text,
        jd_text=jd_text,
        resume_keywords=resume_keywords,
        jd_keywords=jd_keywords,
        selected_questions=questions,
        retrieval_mode=retrieval_mode,
    )
    for q in questions:
        session.records.append(QuestionRecord(question=q))
    return session


def get_interviewer_response(session: InterviewSession, user_answer: str = "") -> str:
    if session.is_finished:
        return json.dumps({
            "action": "end_topic",
            "message": "面试已结束。请查看评估报告。",
            "score": None, "feedback": None
        }, ensure_ascii=False)

    record = session.current_record
    if record is None:
        session.is_finished = True
        return json.dumps({
            "action": "end_topic",
            "message": "所有题目已完成，面试结束！请生成评估报告。",
            "score": None, "feedback": None
        }, ensure_ascii=False)

    jd_summary = f"目标岗位关键词: {', '.join(session.jd_keywords[:10])}"
    if session.jd_text:
        jd_summary += f"\n\n{session.jd_text[:300]}"

    resume_summary = f"简历关键词: {', '.join(session.resume_keywords[:15])}"
    if session.resume_text:
        resume_summary += f"\n\n{session.resume_text[:500]}"

    system_prompt = SYSTEM_PROMPT_INTERVIEWER.format(
        jd_summary=jd_summary,
        current_question=record.question.title,
        answer_key_points=record.question.answer_key_points,
        resume_summary=resume_summary,
    )

    messages = [{"role": "system", "content": system_prompt}]
    for exchange in record.exchanges:
        role = "assistant" if exchange["role"] == "interviewer" else "user"
        messages.append({"role": role, "content": exchange["content"]})

    if user_answer:
        messages.append({"role": "user", "content": user_answer})
        record.exchanges.append({"role": "candidate", "content": user_answer})

    response = chat_completion(messages, temperature=0.7)
    result = _parse_response(response)

    action = result.get("action", "follow_up")
    message = result.get("message", "请继续回答。")
    score = result.get("score")
    feedback = result.get("feedback")

    record.exchanges.append({"role": "interviewer", "content": message})

    if score is not None:
        record.score = score
        record.feedback = feedback or ""

    if action in ("ask_new", "end_topic"):
        session.current_followup_count = 0
        session.current_question_idx += 1
        if session.current_question_idx >= len(session.selected_questions):
            session.is_finished = True
            message += "\n\n🎉 所有题目已完成，面试结束！请生成评估报告。"
    elif action == "follow_up":
        session.current_followup_count += 1
        if session.current_followup_count >= MAX_FOLLOWUPS:
            session.current_followup_count = 0
            session.current_question_idx += 1
            if session.current_question_idx >= len(session.selected_questions):
                session.is_finished = True
                message += "\n\n🎉 面试结束！"
            else:
                next_q = session.selected_questions[session.current_question_idx]
                message += f"\n\n好的，下一个问题：\n\n**{next_q.title}**"
                new_record = session.records[session.current_question_idx]
                new_record.exchanges.append({"role": "interviewer", "content": f"下一个问题：{next_q.title}"})

    return json.dumps({
        "action": action, "message": message,
        "score": score, "feedback": feedback
    }, ensure_ascii=False)


def _parse_response(response: str) -> dict:
    import re
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {"action": "follow_up", "message": response, "score": None, "feedback": None}
