"""能力雷达图生成"""
import io
import base64
import numpy as np
from src.config import SCORE_DIMENSIONS

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False


def generate_radar_chart(dimensions: dict) -> str | None:
    """
    生成雷达图，返回 base64 编码的 PNG 图片路径。
    如果 matplotlib 不可用，返回 None。
    """
    if not _MPL_AVAILABLE:
        return None

    # 准备数据
    labels = []
    scores = []
    for key, config in SCORE_DIMENSIONS.items():
        labels.append(config["label"])
        dim_data = dimensions.get(key, {})
        scores.append(dim_data.get("score", 0))

    num_vars = len(labels)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    scores_plot = scores + [scores[0]]
    angles += [angles[0]]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    # 设置中文字体
    try:
        font = FontProperties(fname="/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc")
    except Exception:
        font = FontProperties()

    ax.fill(angles, scores_plot, color="dodgerblue", alpha=0.25)
    ax.plot(angles, scores_plot, color="dodgerblue", linewidth=2)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontproperties=font, fontsize=11)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(["2", "4", "6", "8", "10"], fontsize=9)
    ax.set_title("能力画像雷达图", fontproperties=font, fontsize=14, pad=20)

    # 标注分数
    for angle, score, label in zip(angles[:-1], scores, labels):
        ax.annotate(f"{score}", xy=(angle, score), fontsize=10, ha="center",
                    color="darkblue", fontweight="bold")

    plt.tight_layout()

    # 保存为临时文件
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # 保存到临时文件并返回路径
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    tmp.write(buf.read())
    tmp.close()
    return tmp.name


def format_score_table(dimensions: dict) -> str:
    """格式化为 Markdown 表格"""
    lines = ["| 维度 | 得分 | 评语 |", "|------|------|------|"]
    for key, config in SCORE_DIMENSIONS.items():
        dim_data = dimensions.get(key, {})
        score = dim_data.get("score", 0)
        comment = dim_data.get("comment", "")
        bar = "█" * score + "░" * (10 - score)
        lines.append(f"| {config['label']} | {bar} {score}/10 | {comment} |")
    return "\n".join(lines)
