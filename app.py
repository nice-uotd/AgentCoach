"""AgentCoach —— 面向 AI 工程实习的模拟面试与能力评估系统"""
import json
import gradio as gr

from src.config import MOCK_MODE, MAX_QUESTIONS
from src.matching import match_questions, load_jd_templates
from src.interview_agent import InterviewSession, start_interview, get_interviewer_response
from src.evaluation import generate_evaluation, compute_weighted_score
from src.radar_chart import generate_radar_chart, format_score_table
from src.question_bank import get_question_bank

# 全局会话
_sessions: dict[str, InterviewSession] = {}

def _get_session() -> InterviewSession | None:
    return _sessions.get("default")


# ==================== Tab 1: 简历 & JD 输入 ====================

def submit_resume_jd(resume_text: str, jd_choice: str, custom_jd: str) -> tuple[str, str]:
    if not resume_text or not resume_text.strip():
        return "❌ 请输入简历内容", ""

    # 确定 JD 文本
    jd_text = ""
    if jd_choice == "自定义 JD":
        jd_text = custom_jd
    else:
        templates = load_jd_templates()
        for t in templates:
            if t["title"] == jd_choice:
                jd_text = f"{t['title']} - {t['description']}\n要求: {'; '.join(t['requirements'])}"
                break

    if not jd_text:
        return "❌ 请选择岗位方向或输入自定义 JD", ""

    # 双匹配选题
    questions, meta = match_questions(resume_text, jd_text, top_k=MAX_QUESTIONS)

    if not questions:
        return "❌ 题库匹配失败，请检查输入", ""

    # 创建 session
    session = start_interview(
        resume_text=resume_text,
        jd_text=jd_text,
        questions=questions,
        resume_keywords=meta["resume_keywords"],
        jd_keywords=meta["jd_keywords"],
        retrieval_mode=meta["retrieval_mode"],
    )
    _sessions["default"] = session

    # 展示结果
    kw_text = "、".join(meta["resume_keywords"][:10])
    jd_kw_text = "、".join(meta["jd_keywords"][:8])
    q_text = "\n".join(
        f"  {i+1}. [{q.category}|{q.difficulty}] {q.title}"
        for i, q in enumerate(questions)
    )

    status = f"""✅ 匹配完成！

**简历关键词**: {kw_text}
**JD 关键词**: {jd_kw_text}
**检索模式**: {meta['retrieval_mode']}

**匹配到 {len(questions)} 道面试题**:
{q_text}

👉 请切换到「🎤 模拟面试」Tab 开始面试！"""

    mode_info = "🟡 Mock 模式" if MOCK_MODE else "🟢 API 模式"
    return status, mode_info


# ==================== Tab 2: 模拟面试 ====================

def start_mock_interview():
    session = _get_session()
    if session is None:
        return [{"role": "assistant", "content": "❌ 请先在「📝 简历 & JD」Tab 提交信息"}], "未开始"

    first_q = session.selected_questions[0]
    opening = (
        f"你好！我是今天的面试官。\n\n"
        f"我看了你的简历，接下来围绕目标岗位对你进行技术面试。\n\n"
        f"第一个问题：\n\n**{first_q.title}**"
    )

    record = session.records[0]
    record.exchanges.append({"role": "interviewer", "content": opening})
    session.chat_history = [{"role": "assistant", "content": opening}]
    return session.chat_history, session.progress_text


def submit_answer(user_input: str, chat_history: list):
    session = _get_session()
    if session is None:
        return [{"role": "assistant", "content": "❌ 请先提交简历"}], "未开始", ""

    if not user_input or not user_input.strip():
        return chat_history, session.progress_text, ""

    chat_history.append({"role": "user", "content": user_input})

    response_json = get_interviewer_response(session, user_input)
    try:
        result = json.loads(response_json)
        message = result.get("message", "")
    except (json.JSONDecodeError, TypeError):
        message = response_json

    chat_history.append({"role": "assistant", "content": message})
    session.chat_history = chat_history

    hint = ""
    if session.is_finished:
        hint = "🎉 面试结束！请切换到「📊 能力画像」Tab 生成报告。"

    return chat_history, session.progress_text, hint


# ==================== Tab 3: 能力画像 ====================

def generate_report():
    session = _get_session()
    if session is None:
        return "❌ 请先完成面试", None

    evaluation = generate_evaluation(session)
    dimensions = evaluation.get("dimensions", {})

    # 生成雷达图
    chart_path = generate_radar_chart(dimensions)

    # 生成文字报告
    overall = evaluation.get("overall_score", 0)
    weighted = compute_weighted_score(dimensions)

    report = f"# 📊 能力画像评估报告\n\n"
    report += f"## 综合评分: {overall}/10 (加权: {weighted}/10)\n\n"

    # 维度表格
    report += "## 六维能力评分\n\n"
    report += format_score_table(dimensions)
    report += "\n\n"

    # 优点
    strengths = evaluation.get("strengths", [])
    if strengths:
        report += "## ✅ 表现亮点\n\n"
        for s in strengths:
            report += f"- {s}\n"
        report += "\n"

    # 薄弱点
    weaknesses = evaluation.get("weaknesses", [])
    if weaknesses:
        report += "## ⚠️ 薄弱点\n\n"
        for s in weaknesses:
            report += f"- {s}\n"
        report += "\n"

    # 遗漏知识点
    missed = evaluation.get("missed_points", [])
    if missed:
        report += "## 🔑 遗漏知识点\n\n"
        for s in missed:
            report += f"- {s}\n"
        report += "\n"

    # 推荐复习路径
    study = evaluation.get("study_path", [])
    if study:
        report += "## 📚 推荐复习路径\n\n"
        for i, s in enumerate(study, 1):
            report += f"{i}. {s}\n"
        report += "\n"

    # 下轮建议
    suggestions = evaluation.get("next_round_suggestions", [])
    if suggestions:
        report += "## 🎯 下一轮模拟面试建议\n\n"
        for s in suggestions:
            report += f"- {s}\n"
        report += "\n"

    # 题目回顾
    report += "## 📖 题目得分回顾\n\n"
    for i, record in enumerate(session.records):
        if record.exchanges:
            score_text = f"{record.score}/10" if record.score else "未评分"
            report += f"**Q{i+1}** [{record.question.category}] {record.question.title}\n"
            report += f"- 得分: {score_text}\n"
            if record.feedback:
                report += f"- 反馈: {record.feedback}\n"
            report += "\n"

    return report, chart_path


# ==================== Tab 4: 评测数据 ====================

def show_eval_results():
    """展示预计算的 BM25 vs Hybrid 评测结果"""
    import os
    eval_path = os.path.join(os.path.dirname(__file__), "eval", "eval_results.json")
    if not os.path.exists(eval_path):
        return "⚠️ 评测尚未运行。请执行 `python eval/run_eval.py` 生成结果。"

    with open(eval_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    report = "# 🧪 BM25 vs Hybrid 检索评测\n\n"
    report += f"**测试集**: {data.get('num_resumes', 0)} 份简历 × {data.get('num_jds', 0)} 个岗位方向\n"
    report += f"**题库规模**: {data.get('num_questions', 0)} 道\n\n"

    report += "## 选题相关性对比\n\n"
    report += "| 指标 | BM25 Only | Hybrid (BM25+向量+RRF) | 提升 |\n"
    report += "|------|-----------|------------------------|------|\n"

    metrics = data.get("metrics", {})
    for metric_name, values in metrics.items():
        bm25_val = values.get("bm25", 0)
        hybrid_val = values.get("hybrid", 0)
        diff = hybrid_val - bm25_val
        sign = "+" if diff > 0 else ""
        report += f"| {metric_name} | {bm25_val:.2f} | {hybrid_val:.2f} | {sign}{diff:.2f} |\n"

    report += "\n## 结论\n\n"
    for c in data.get("conclusions", []):
        report += f"- {c}\n"

    return report


# ==================== 构建 Gradio App ====================

def create_app() -> gr.Blocks:
    # 加载 JD 模板
    jd_templates = load_jd_templates()
    jd_choices = [t["title"] for t in jd_templates] + ["自定义 JD"]

    with gr.Blocks(title="AgentCoach - AI 模拟面试与能力评估") as app:
        gr.Markdown(
            "# 🎯 AgentCoach\n"
            "**面向 AI 工程实习的模拟面试与能力评估系统**\n\n"
            "简历 + JD 双匹配 → Hybrid 检索选题 → ReAct 面试官 → 6维能力画像\n\n"
            f"{'🟡 Mock 演示模式' if MOCK_MODE else '🟢 已连接 LLM API'} | "
            f"题库: {len(get_question_bank().questions)} 道 | "
            f"检索: {get_question_bank().retrieval_mode}"
        )

        with gr.Tabs():
            # ---- Tab 1 ----
            with gr.Tab("📝 简历 & JD"):
                gr.Markdown("### 上传简历 & 选择岗位\n系统根据**简历 + 岗位 JD**双匹配个性化面试题。")
                with gr.Row():
                    with gr.Column(scale=3):
                        resume_input = gr.Textbox(
                            label="简历内容（项目经历为主）",
                            placeholder="粘贴你的项目经历、技术栈...\n\n示例：\n基于 RAG 的知识问答系统，负责检索管道设计和 Agent 编排。使用 Milvus + BM25 多路召回 + RRF 融合...",
                            lines=10,
                        )
                    with gr.Column(scale=2):
                        jd_dropdown = gr.Dropdown(
                            choices=jd_choices,
                            label="目标岗位",
                            value=jd_choices[0],
                        )
                        custom_jd_input = gr.Textbox(
                            label="自定义 JD（选择「自定义 JD」时填写）",
                            placeholder="粘贴岗位要求...",
                            lines=5,
                            visible=True,
                        )
                        submit_btn = gr.Button("🚀 匹配面试题", variant="primary")
                        mode_display = gr.Textbox(label="运行模式", interactive=False)

                status_output = gr.Markdown()
                submit_btn.click(
                    fn=submit_resume_jd,
                    inputs=[resume_input, jd_dropdown, custom_jd_input],
                    outputs=[status_output, mode_display],
                )

            # ---- Tab 2 ----
            with gr.Tab("🎤 模拟面试"):
                gr.Markdown("### 模拟面试对话\n面试官根据回答质量决定追问或换题。每题最多追问 3 次。")
                with gr.Row():
                    start_btn = gr.Button("▶️ 开始面试", variant="primary")
                    progress_display = gr.Textbox(label="进度", interactive=False, scale=2)

                chatbot = gr.Chatbot(label="面试对话", height=450)

                with gr.Row():
                    user_input = gr.Textbox(
                        label="你的回答", placeholder="输入回答...", lines=3, scale=4
                    )
                    answer_btn = gr.Button("📤 提交", variant="primary", scale=1)

                finish_hint = gr.Markdown("")

                start_btn.click(fn=start_mock_interview, outputs=[chatbot, progress_display])
                answer_btn.click(
                    fn=submit_answer,
                    inputs=[user_input, chatbot],
                    outputs=[chatbot, progress_display, finish_hint],
                ).then(fn=lambda: "", outputs=[user_input])
                user_input.submit(
                    fn=submit_answer,
                    inputs=[user_input, chatbot],
                    outputs=[chatbot, progress_display, finish_hint],
                ).then(fn=lambda: "", outputs=[user_input])

            # ---- Tab 3 ----
            with gr.Tab("📊 能力画像"):
                gr.Markdown("### 6维能力画像\n面试结束后生成能力雷达图 + 薄弱点分析 + 复习路径。")
                report_btn = gr.Button("📊 生成能力画像", variant="primary")
                with gr.Row():
                    radar_image = gr.Image(label="能力雷达图", scale=1)
                    report_output = gr.Markdown(label="评估报告", scale=2)

                report_btn.click(fn=generate_report, outputs=[report_output, radar_image])

            # ---- Tab 4 ----
            with gr.Tab("🧪 评测实验"):
                gr.Markdown("### BM25 vs Hybrid 检索评测\n10份模拟简历 × 3个岗位方向，对比选题相关性。")
                eval_btn = gr.Button("📈 查看评测结果", variant="primary")
                eval_output = gr.Markdown()
                eval_btn.click(fn=show_eval_results, outputs=[eval_output])

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
