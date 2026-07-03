"""LLM 客户端封装（支持 DeepSeek API + Mock 模式）"""
import json
import random
from openai import OpenAI
from src.config import OPENAI_API_BASE, OPENAI_API_KEY, OPENAI_MODEL, MOCK_MODE


def get_client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_API_BASE)


def chat_completion(messages: list[dict], temperature: float = 0.7, max_tokens: int = 1024) -> str:
    if MOCK_MODE:
        return _mock_response(messages)
    client = get_client()
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message.content or ""


def _mock_response(messages: list[dict]) -> str:
    last_msg = messages[-1]["content"] if messages else ""
    system_msg = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""

    if "简历解析" in system_msg or "提取关键" in system_msg:
        return json.dumps({
            "projects": ["RAG 检索系统", "多Agent协作平台", "LLM应用开发"],
            "skills": ["Python", "LangChain", "向量数据库", "Prompt Engineering", "FastAPI"],
            "keywords": ["检索增强生成", "Agent编排", "Function Calling", "知识图谱", "ReAct"]
        }, ensure_ascii=False)

    if "JD 解析" in system_msg or "岗位要求" in system_msg:
        return json.dumps({
            "required_skills": ["RAG", "Agent", "向量数据库", "Python"],
            "bonus_skills": ["Function Calling", "MCP", "微调"],
            "focus_areas": ["检索架构", "Agent编排", "工程化能力"]
        }, ensure_ascii=False)

    if "面试评估专家" in system_msg or "能力画像" in system_msg:
        return _mock_evaluation()

    if "面试官" in system_msg:
        return _mock_interviewer(last_msg)

    if "评估" in system_msg or "能力画像" in system_msg:
        return _mock_evaluation()

    return json.dumps({"message": "Mock 模式回复"}, ensure_ascii=False)


def _mock_interviewer(user_answer: str) -> str:
    tech_keywords = ["向量", "BM25", "RRF", "rerank", "embedding", "召回", "Milvus",
                     "检索", "分块", "chunk", "语义", "融合", "Agent", "ReAct", "Function"]
    kw_count = sum(1 for kw in tech_keywords if kw in user_answer)

    if kw_count >= 4 or len(user_answer) > 200:
        return json.dumps({
            "action": "ask_new",
            "message": random.choice([
                "回答得很全面，技术细节到位。下一个问题：\n\n**你的 Agent 系统是如何做错误恢复和熔断降级的？**",
                "很好，能看出你对这块有实际经验。换个方向：\n\n**你是怎么做模型幻觉防控的？**",
            ]),
            "score": random.randint(7, 9),
            "feedback": "回答包含具体技术细节和量化数据，覆盖核心要点。"
        }, ensure_ascii=False)
    elif kw_count >= 2 or len(user_answer) > 100:
        return json.dumps({
            "action": "follow_up",
            "message": random.choice([
                "思路不错，但能更具体吗？比如你们的 Recall@10 具体是多少？做过什么对比实验？",
                "理解你的方向了。能用一个具体 case 走一遍流程吗？用户输入 X，系统怎么一步步返回结果？",
            ]),
            "score": random.randint(5, 7),
            "feedback": "方向正确但缺少具体数据和细节。"
        }, ensure_ascii=False)
    else:
        return json.dumps({
            "action": "follow_up",
            "message": random.choice([
                "回答比较笼统。能具体说下你们用什么向量数据库？分块策略是什么？embedding 模型选的哪个？",
                "太宽泛了。请用 STAR 法则描述：当时场景是什么？你做了什么？结果怎样？",
            ]),
            "score": random.randint(3, 5),
            "feedback": "回答空泛，缺少技术深度。"
        }, ensure_ascii=False)


def _mock_evaluation() -> str:
    return json.dumps({
        "overall_score": 7.0,
        "dimensions": {
            "technical_depth": {"score": 7, "comment": "对核心技术有理解，部分回答缺少量化数据。"},
            "engineering_exp": {"score": 6, "comment": "有一定工程意识，但生产经验描述不够深入。"},
            "clarity": {"score": 8, "comment": "表达逻辑清晰，结构化组织较好。"},
            "star_completeness": {"score": 6, "comment": "部分回答缺少 Result 量化。"},
            "system_design": {"score": 7, "comment": "能描述系统架构，但扩展性讨论不足。"},
            "rag_agent_skill": {"score": 7, "comment": "RAG 理解准确，Agent 编排有基础认知。"}
        },
        "strengths": [
            "对 RAG 架构的整体理解准确",
            "能清晰描述技术选型理由",
            "表达结构化，逻辑连贯"
        ],
        "weaknesses": [
            "量化数据不足（缺少准确率、延迟等具体数字）",
            "STAR 法则运用不完整，Result 部分薄弱",
            "对生产环境的故障处理经验描述有限"
        ],
        "missed_points": [
            "RRF 融合公式 score=Σ1/(k+rank_i) 的具体解释",
            "模型路由的分级策略和成本节省数据",
            "熔断器三状态切换的具体阈值设计"
        ],
        "study_path": [
            "深入理解 HNSW/IVF 索引原理，准备能画图解释",
            "准备 2-3 个带具体数据的性能优化案例",
            "练习用 STAR 法则组织每个回答，确保有量化 Result"
        ],
        "next_round_suggestions": [
            "重点练习故障处理类题目（死循环检测、幻觉控制）",
            "每个回答准备至少 2 个量化数据点",
            "尝试系统设计类题目的白板演练"
        ]
    }, ensure_ascii=False)
