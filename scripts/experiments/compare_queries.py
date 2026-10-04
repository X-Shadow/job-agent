"""检查：检索结果冒烟检查（**不是对照实验**）

背景：切换 embedding 后端、重建索引、改检索逻辑之后，需要一个快速手段确认
      "Top-5 没崩"。

跑法：python scripts/experiments/compare_queries.py

结论：**本脚本没有 A/B 结论**——它只展示当前配置下的一侧结果，没有对照组。
      要做对照（同进程跑两侧），用同目录的 2026-10-04_instruction_ab.py。
      它支撑的是 docs/runtime_log.md 中 2026-10-04「切换 embedding 后端」
      一节的 6 组排名对比；该对比已标注局限：两轮之间改过 experience.json，
      不是干净的 A/B。

对应决策记录：docs/decisions.md「Embedding 后端：智谱 API → 本地 bge-small-zh-v1.5」
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

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
