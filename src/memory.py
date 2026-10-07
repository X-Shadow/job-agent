"""多轮上下文管理：历史过长时压缩（compaction）。

为什么不把全部历史原样塞回去：
- 每轮的 prompt 会随对话轮数线性增长，成本与延迟一起涨
- 无关的旧内容会稀释注意力——和检索里"只喂相关经历"是同一个思路

面试里能说清"我为什么对历史做 compaction、压缩前后差多少"，就是 context engineering
的实证。所以这里不只压缩，还**如实报告压缩前后的规模差异**（CompactionReport）。

token 数是**估算**：DeepSeek 不公开分词器，按中文 0.6 token/字、
非中文 4 字符≈1 词、1.3 token/词估算。真实用量以 API 返回的 usage 为准。
"""
import logging

import llm
import schemas

DEFAULT_MAX_TOKENS = 3000
DEFAULT_KEEP_RECENT = 4

COMPACT_PROMPT = """把下面这段多轮对话压缩成一段摘要，供后续轮次继续使用。

要求：
1. 保留：用户的目标与约束、已确认的事实、已达成的结论、未解决的问题
2. 保留具体的名称与数字（岗位名、公司名、工具名、指标值）
3. 丢弃：寒暄、重复表述、已被推翻的中间结论、工具调用的原始输出
4. 用第三人称陈述，不要写成对话形式，400 字以内"""


def estimate_tokens(text: str) -> int:
    """粗估 token 数（中文 0.6 token/字；非中文按 4 字符≈1 词、1.3 token/词）。"""
    if not text:
        return 0
    zh = sum(1 for ch in text if "一" <= ch <= "鿿")
    other = len(text) - zh
    return int(zh * 0.6 + (other / 4) * 1.3)


def history_tokens(messages: list[dict]) -> int:
    """估算一整段消息历史的 token 数。"""
    return sum(estimate_tokens(m.get("content") or "") for m in messages)


def _cut_index(convo: list[dict], keep_recent: int) -> int:
    """算出"最近 keep_recent 条"从哪里开始。

    注意：不能把 role=tool 的消息切成窗口的第一条——tool 消息必须紧跟在
    带 tool_calls 的 assistant 消息之后，切错了 API 会直接拒绝请求。
    """
    cut = max(0, len(convo) - keep_recent)
    while cut < len(convo) and convo[cut].get("role") == "tool":
        cut += 1
    return cut


def compact_history(
    messages: list[dict],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    keep_recent: int = DEFAULT_KEEP_RECENT,
) -> tuple[list[dict], schemas.CompactionReport]:
    """历史超阈值时压缩，返回 (新历史, 观测记录)。

    保留开头的 system 与最近 keep_recent 条，中间部分交给 LLM 摘要成一条 system。
    未超阈值时原样返回，report.triggered=False。

    返回元组而不是只返回列表：压缩前后的 token 差异是评估素材，
    必须能被调用方（界面 / 实验脚本）拿到并记录。
    """
    report = schemas.CompactionReport(before_tokens=history_tokens(messages))

    if report.before_tokens <= max_tokens or len(messages) <= keep_recent:
        report.after_tokens = report.before_tokens
        return messages, report

    has_system = bool(messages) and messages[0].get("role") == "system"
    head = messages[:1] if has_system else []
    convo = messages[1:] if has_system else messages

    cut = _cut_index(convo, keep_recent)
    old, recent = convo[:cut], convo[cut:]
    if not old:                      # 没有可压缩的部分（都被安全边界挡回去了）
        report.after_tokens = report.before_tokens
        return messages, report

    transcript = "\n".join(
        f"{m.get('role')}: {(m.get('content') or '')[:800]}" for m in old)
    summary = llm.chat(
        [
            {"role": "system", "content": COMPACT_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.3,
    )

    new_messages = head + [{"role": "system", "content": f"【历史摘要】\n{summary}"}] + recent

    report.triggered = True
    report.summarized_messages = len(old)
    report.kept_messages = len(recent)
    report.after_tokens = history_tokens(new_messages)

    logging.info(
        "历史压缩: %s → %s tokens（估算），摘要 %s 条 / 保留 %s 条",
        report.before_tokens, report.after_tokens, len(old), len(recent),
    )
    return new_messages, report
