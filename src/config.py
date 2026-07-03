"""配置管理"""
import os
from dotenv import load_dotenv

load_dotenv()

# API 配置
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://dashscope.aliyuncs.com/compatible-mode/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "deepseek-v3")

# Mock 模式
MOCK_MODE = os.getenv("MOCK_MODE", "false").lower() == "true" or not OPENAI_API_KEY

# 面试配置
MAX_QUESTIONS = 5
MAX_FOLLOWUPS = 3

# 评估维度与权重
SCORE_DIMENSIONS = {
    "technical_depth": {"label": "技术深度", "weight": 0.20},
    "engineering_exp": {"label": "工程经验", "weight": 0.20},
    "clarity": {"label": "表达清晰度", "weight": 0.15},
    "star_completeness": {"label": "STAR 完整性", "weight": 0.15},
    "system_design": {"label": "系统设计能力", "weight": 0.15},
    "rag_agent_skill": {"label": "RAG/Agent 专项", "weight": 0.15},
}

# 岗位方向
JOB_DIRECTIONS = ["AI Agent 实习生", "RAG 工程师", "LLM 应用开发"]

# Hybrid 检索配置
RRF_K = 60
BM25_WEIGHT = 0.4
VECTOR_WEIGHT = 0.6
VECTOR_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
