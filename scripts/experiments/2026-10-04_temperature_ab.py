"""
实验：temperature 降到 0 能否消除 parse_jd 的条数波动

背景：上面那个稳定性实验发现 must_have 条数会变。第一反应是"采样随机"，
      如果降到 temperature=0 能消除，那就是成本最低的修法。
      设计成同一轮会话内跑两种温度，控制模型与时间变量。

跑法：python scripts/experiments/2026-10-04_temperature_ab.py

结论：**无效**。temperature=0.1 → must_have [5, 6, 7]（范围 3）；
      temperature=0.0 → [7, 8, 4]（范围 5，反而更大）。
      说明不确定性并非来自采样随机，而是「这句话该不该拆」本身就是语义歧义。
      累计 9 次观测（含稳定性实验），must_have 落在 4~8 条。

对应决策记录：docs/decisions.md「parse_jd 输出不稳定 → 引入运行存档」（待补写）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import llm  # noqa: E402
import schemas  # noqa: E402
import tools  # noqa: E402
from config import DATA_DIR  # noqa: E402

REPEATS = 3
TEMPERATURES = (0.1, 0.0)

if __name__ == "__main__":
    jd_text = (DATA_DIR / "test_jd.txt").read_text(encoding="utf-8")

    for temp in TEMPERATURES:
        print(f"--- temperature={temp} ---")
        counts = []
        for i in range(1, REPEATS + 1):
            info = llm.chat_structured(
                schemas.JDInfo,
                [
                    {"role": "system", "content": tools.PARSE_JD_PROMPT},
                    {"role": "user", "content": jd_text},
                ],
                temperature=temp,
            )
            counts.append(len(info.must_have))
            print(f"    第 {i} 次: must_have={len(info.must_have)}  "
                  f"keywords={len(info.keywords)}")
        print(f"    must_have 取值: {counts}  波动范围: {max(counts) - min(counts)}\n")
