"""检索冒烟检查：跑一组固定查询，打印当前配置下的 Top-5。

定位：这是**冒烟工具，不是对照实验**——只展示当前一侧的结果，
      不构成 A/B（没有对照组）。要做对照（同进程跑两侧），
      用 scripts/experiments/ 下的脚本。

用途：切换 embedding 后端、重建索引、改检索逻辑之后，一眼确认 Top-5 没崩。
      距离数值在日志里输出（logging 打到 stderr）。

运行: python scripts/compare_queries.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import tools  # noqa: E402
from config import (  # noqa: E402
    EMBED_BACKEND,
    EMBED_MODEL,
    LOCAL_EMBED_MODEL,
    USE_QUERY_INSTRUCTION,
)

QUERIES = [
    "Agent 开发经验",
    "FastAPI 后端",
    "RAG 检索",
    "嵌入式 单片机",
    "Git 版本控制",
    "用 Python 做 Web 后端接口",
]

if __name__ == "__main__":
    model_name = LOCAL_EMBED_MODEL if EMBED_BACKEND == "local" else EMBED_MODEL
    print(f"后端={EMBED_BACKEND}  模型={model_name}  查询指令={USE_QUERY_INSTRUCTION}\n")
    for q in QUERIES:
        print(f"=== {q} ===")
        for e in tools.search_experience(q):
            print(f"  {e.id} | {e.title}")
        print()
