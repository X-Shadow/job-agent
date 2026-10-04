"""对比检索效果：跑一组固定查询，打印 Top-5（距离见日志）。

用途：
- 切换 embedding 后端（本地 bge ↔ 智谱 API）后对比排名变化
- 开关查询指令（USE_QUERY_INSTRUCTION）前后对比

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
