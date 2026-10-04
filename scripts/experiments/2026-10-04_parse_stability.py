"""
实验：parse_jd 对同一份 JD 的输出稳定性

背景：同一份 JD（749 字符，内容未改）在不同运行中解析出的 must_have 条数不一致。
      must_have 条数决定下游逐条判定的条目集合，会让"匹配一致率"这类
      跨运行指标混入解析噪声——测到的可能是解析差异，而不是判定差异。

跑法：python scripts/experiments/2026-10-04_parse_stability.py

结论：3 次运行 must_have = 6 / 4 / 6，keywords = 34 / 32 / 30。
      波动来自「逐条拆开，不要把多条合并成一条」这条规则的粒度缺乏明确边界：
      「了解大语言模型基本使用方式，熟悉 Prompt Engineering…」有时合为 1 条、有时拆成 2 条。
      事实核对：该 JD 任职要求中不带"优先"字样的是 4 条 → 4 条才与人工判断一致，6 条属拆得过细。
      原始数据见 docs/runtime_log.md（2026-10-04 节）。

对应决策记录：docs/decisions.md「parse_jd 输出不稳定 → 引入运行存档」（待补写）
"""
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import tools  # noqa: E402
from config import DATA_DIR  # noqa: E402

REPEATS = 3

if __name__ == "__main__":
    jd_text = (DATA_DIR / "test_jd.txt").read_text(encoding="utf-8")
    print(f"JD 原文长度: {len(jd_text)} 字符\n")

    counts = []
    for i in range(1, REPEATS + 1):
        info = tools.parse_jd(jd_text)
        counts.append(len(info.must_have))
        print(f"第 {i} 次: must_have={len(info.must_have)}  "
              f"nice_to_have={len(info.nice_to_have)}  keywords={len(info.keywords)}")
        for m in info.must_have:
            print(f"     - {m[:45]}")
        print()

    print(f"must_have 取值: {counts}")
    print(f"波动范围: {max(counts) - min(counts)}"
          f"  出现次数: {dict(Counter(counts))}")
