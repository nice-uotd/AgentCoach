---
title: AgentCoach
emoji: 🎯
colorFrom: blue
colorTo: purple
sdk: gradio
app_file: app.py
pinned: false
license: mit
short_description: 面向AI工程实习的模拟面试与能力评估系统
---



# 🎯 AgentCoach

**面向 AI 工程实习的模拟面试与能力评估系统**

基于 Hybrid 检索（BM25 + 向量语义 + RRF 融合）进行个性化选题，通过 LLM 决策驱动的多轮追问 Agent 模拟面试，生成 6 维能力画像雷达图。

## 核心特性

| 特性 | 说明 |
|------|------|
| 📚 92 道结构化题库 | 7 大类别，含参考答案 + 评估标准 + 追问 |
| 🎯 JD + 简历双匹配 | 根据目标岗位 + 个人背景个性化选题 |
| 🔍 Hybrid 检索 | BM25 关键词 + 向量语义 + RRF 融合排序 |
| 🤖 LLM 决策面试官 | 对照参考答案智能追问，最多 3 轮深挖 |
| 📊 6 维能力画像 | 技术深度/工程经验/表达/STAR/系统设计/专项能力 |
| 🧪 评测实验 | 10 简历 × 3 岗位，BM25 vs Hybrid 对比 |

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API（可选）

```bash
cp .env.example .env
# 编辑 .env 填入 API Key
# 不配置则自动进入 Mock 演示模式
```

### 3. 启动

```bash
python app.py
# 访问 http://localhost:7860
```

### Docker 部署

```bash
docker build -t agentcoach .
docker run -p 7860:7860 agentcoach
```

## 系统架构

```
用户输入(简历+JD)
    │
    ▼
┌──────────────────────────┐
│  简历解析 + JD 解析 (LLM) │
└──────────────────────────┘
    │ 关键词
    ▼
┌──────────────────────────┐
│  Hybrid 检索引擎          │
│  BM25 ──┐                │
│          ├─→ RRF 融合排序  │
│  向量 ──-┘                │
└──────────────────────────┘
    │ Top-5 题目
    ▼
┌──────────────────────────┐
│  状态机驱动的面试 Agent    │
│  (LLM决策: 追问/评分/换题) │
└──────────────────────────┘
    │ 面试记录
    ▼
┌──────────────────────────┐
│  6 维能力评估              │
│  雷达图 + 薄弱点 + 学习路径 │
└──────────────────────────┘
```

## 评测实验

```bash
python -m eval.run_eval
```

对比 BM25 vs Hybrid 在 30 组（10简历×3岗位）测试中的选题相关性。

## 技术栈

- **检索**: rank-bm25 + sentence-transformers + RRF
- **LLM**: OpenAI 兼容 API (DeepSeek/GPT)
- **前端**: Gradio
- **可视化**: matplotlib (雷达图)
- **部署**: Docker / HuggingFace Spaces

## 项目定位

与 [PaperFlow](https://github.com/nice-uotd/PaperFlow) 形成技术作品集：
- **[PaperFlow](https://huggingface.co/spaces/nice-uotd/PaperFlow)**: 科研文献 RAG + Agent 工具调度
- **AgentCoach**: AI 工程岗位模拟面试 + 能力评估

两个项目共享 Hybrid Retrieval 技术路线，展示 RAG/Agent 工程能力和 AI 产品化能力。

## License

MIT
