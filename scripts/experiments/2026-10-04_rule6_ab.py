"""
实验：改写提示词规则 6（不得删除或淡化诚实标注）是否有效

背景：rewrite_section 曾建议"删掉 exp_008 的『课程代码未跟跑』这种主动减分项"。
      现有规则堵的是「不许新增不存在的事实」，没堵住「删掉真实的限定条件」——
      后者不违反"不新增"，所以规则拦不住，但同样是失真，且更隐蔽（读者看不出少了什么）。
      为此给 REWRITE_PROMPT 加了第 6 条。

跑法：python scripts/experiments/2026-10-04_rule6_ab.py

结论：**未能证明**——原始失效模式没有复现。两侧的建议正文都没有出现删除/淡化类表述
      （各 0/5 条），且两侧都保留了「课程代码未跟跑」。原始观察来自另一次运行、
      输入不同（4 条 must_have vs 本次 7 条），不可直接对比。
      可观察到的行为变化：有规则那侧会在 reason 里**显式声明保留**诚实标注
      （"『基础语法』这一诚实标注保留不动"），即效果是"把隐性约束显式化"。
      **副产物（比实验目的更重要）**：rewrite_section 输出方差极大——同一份输入、
      三次运行给出三套主题完全不同的建议，与 parse_jd 属同一类不稳定问题。

测量注意：敏感词判定范围是「建议正文 + 理由」两个字段。若只统计正文、不打印理由，
      命中将无法复核（"删掉它"与"不得删掉它"在计数上无法区分）。本脚本已分别统计并打印。

对应决策记录：docs/decisions.md（待补写：改写工具的选择性隐瞒防护）
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

import schemas  # noqa: E402
import tools  # noqa: E402
from config import DATA_DIR  # noqa: E402

# 判定「删除/淡化」类表述的敏感词
MARKERS = ["删掉", "删除", "不要写", "别写", "自我示弱", "减分"]
# 被观察的诚实标注——两侧都应保留
CAVEAT = "未跟跑"


def latest_run_dir() -> Path:
    """取最近一次运行存档（含 jdinfo.json），保证脚本不依赖固定时间戳。"""
    runs = sorted(p for p in (DATA_DIR / "runs").iterdir() if p.is_dir())
    if not runs:
        sys.exit("data/runs 下没有存档，先用 python src/tools.py 跑一次")
    return runs[-1]


if __name__ == "__main__":
    run_dir = latest_run_dir()
    print(f"使用存档: {run_dir}\n")

    jd_info = schemas.JDInfo.model_validate_json(
        (run_dir / "jdinfo.json").read_text(encoding="utf-8"))
    basics = json.loads(tools.EXPERIENCE_FILE.read_text(encoding="utf-8"))["basics"]
    hits = tools.search_experience(" ".join(jd_info.keywords), k=tools.library_size())

    # 两侧共用同一份匹配结果——这是本实验唯一变量成立的前提
    match = tools.score_match(jd_info, hits, basics)
    print(f"score_match 完成（{len(match.requirement_matches)} 条要求），两侧共用\n")

    with_rule = tools.REWRITE_PROMPT
    without_rule = "\n".join(
        line for line in with_rule.splitlines() if not line.startswith("6. 改写"))
    assert without_rule != with_rule, "规则 6 没被摘掉，检查前缀是否写对"

    for label, prompt in (("【基线·无规则6】", without_rule), ("【有规则6】", with_rule)):
        tools.REWRITE_PROMPT = prompt
        plan = tools.rewrite_section(jd_info, match, hits, basics)

        print("=" * 64)
        print(label, f"共 {len(plan.suggestions)} 条建议")
        in_body, in_reason = [], []
        for i, s in enumerate(plan.suggestions, 1):
            bad_body = [m for m in MARKERS if m in s.rewritten]
            bad_reason = [m for m in MARKERS if m in s.reason]
            if bad_body:
                in_body.append(i)
            if bad_reason:
                in_reason.append(i)
            print(f"\n{i}. [{s.target}]")
            print(f"   建议: {s.rewritten}")
            print(f"   理由: {s.reason}")
            if bad_body:
                print(f"   ⚠ 建议正文含敏感词: {bad_body}")
            if bad_reason:
                print(f"   ⚠ 理由字段含敏感词: {bad_reason}")

        print(f"\n小结: {plan.summary}")
        print(f"→ 建议正文含「删除/淡化」类表述: {len(in_body)} 条 {in_body}")
        print(f"→ 理由字段提及「删掉/删除」: {len(in_reason)} 条 {in_reason}")
        print(f"→ 是否保留「{CAVEAT}」这一诚实标注: "
              f"{any(CAVEAT in s.rewritten for s in plan.suggestions)}")
        print()

    tools.REWRITE_PROMPT = with_rule
