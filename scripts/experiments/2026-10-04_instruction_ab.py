"""
实验：检索查询要不要加 bge 的 query instruction（A/B）

背景：bge 官方给出查询指令「为这个句子生成表示以用于检索相关文章：」，
      但同时说 v1.5 加不加差异很小、"最好在你自己任务上测"。
      文档侧一律不加指令（官方明确约定）。这里只测查询侧加不加。

跑法：python scripts/experiments/2026-10-04_instruction_ab.py

结论：**6 组查询里 Top-5 完全一致的只有 1 组**，但差异**全部落在第 3-5 名**，
      第 1 名只在「FastAPI 后端」这组换过——而那组是全库无真货的凑数组。
      更关键的发现：加指令后距离**系统性上移**，每组每条约 +0.10~+0.27，
      是统一尺度平移而非个别条目变化——副作用是既有阈值（强相关 0.42~0.67 等）全部失效。
      故选择**不加**：没有可辨识的排序改善，却要多维护一个变量并重标一套阈值。
      （与 bge 官方"v1.5 不用指令只有轻微退化"的说法一致。）
      局限：6 条查询、无标注正确答案，"未观察到改善"不等于"证明更差"。

对应决策记录：docs/decisions.md「查询指令：加还是不加」（待补写）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import llm  # noqa: E402
import tools  # noqa: E402

QUERIES = [
    "Agent 开发经验",
    "FastAPI 后端",
    "RAG 检索",
    "嵌入式 单片机",
    "Git 版本控制",
    "用 Python 做 Web 后端接口",
]

if __name__ == "__main__":
    # 同一进程内连续跑两侧：模型实例与索引完全相同，唯一变量是指令开关。
    # 开关在 llm 模块里，运行时改它即可（config 文件不动）。
    results = {}
    for flag in (False, True):
        llm.USE_QUERY_INSTRUCTION = flag
        print(f"\n##### 查询指令={flag}")
        results[flag] = {q: [e.id for e in tools.search_experience(q)] for q in QUERIES}

    print("\n\n===== A/B 对比（Top-5 顺序）=====")
    same = 0
    for q in QUERIES:
        a, b = results[False][q], results[True][q]
        same += a == b
        print(f"\n[{'一致' if a == b else '不同'}] {q}")
        print(f"  不加指令: {a}")
        print(f"  加指令:   {b}")
        if a and b and a[0] != b[0]:
            print(f"  ↑ 第一名变化: {a[0]} → {b[0]}")
    print(f"\nTop-5 完全一致的查询: {same}/{len(QUERIES)}")
