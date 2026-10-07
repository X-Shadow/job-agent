"""Agent 循环：思考 → 选工具 → 执行 → 观察 → 再思考。

这是项目的心脏。前面几天的工具都是在 tools.py 的 __main__ 里按写死的顺序调用的，
这里改成由模型自己决定调哪个工具、调几次、什么时候停。

关键设计（方案 A：会话状态）
    工具的中间结果（JDInfo、检索命中的经历、匹配结果）存在 Session 里，
    模型调用下游工具时不需要把上游结果当参数回传——省 token，模型也无法篡改中间结果。
    代价是隐式顺序依赖，所以 _dispatch 对"前置条件没满足"给人话报错，引导模型自己改正。

工具调用使用原生 function calling（tools= / tool_calls），不是自己解析 JSON。
"""
import json
import logging
import sys
import time
from pathlib import Path

import llm
import memory
import schemas
import tools
from config import DATA_DIR, EXPERIENCE_FILE

# 观察结果回喂给模型时的长度上限——trajectory 里保留截断后的版本即可
MAX_OBSERVATION_CHARS = 4000

SYSTEM_PROMPT = """你是求职助手 Agent。用户会给你一份招聘 JD，或者一个求职相关的请求。

你可以调用工具来完成工作，请一步一步来：
1. 先 parse_jd 解析 JD，再 search_experience 检索候选人的相关经历
2. 然后 score_match 逐条判定匹配，再按用户需要调用 rewrite_section 或 generate_cover_letter
3. 用户要求记录投递时用 track_application；问"投了哪些/进展如何"时用 list_applications

注意：
- 不要跳过前置步骤（没检索经历就做匹配判定，会拿不到依据）
- 工具报错时先读错误信息，按提示补齐前置步骤再重试
- 信息足够时就给出最终答复，不要为了用满工具而多调

回答用中文，简洁具体，不说空话。"""


class Session:
    """一次运行的会话状态（方案 A）。中间结果存这里，工具之间靠它传递。"""

    def __init__(self) -> None:
        self.basics: dict = {}
        self.jd_info: schemas.JDInfo | None = None
        self.hits: list[schemas.Experience] = []
        self.match: schemas.MatchResult | None = None
        self.rewrite: schemas.RewritePlan | None = None
        self.letter: schemas.CoverLetter | None = None
        # 用量累计——由 run_agent 在结束时写入
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.elapsed_ms = 0

    def load_basics(self) -> None:
        with EXPERIENCE_FILE.open(encoding="utf-8") as f:
            self.basics = json.load(f)["basics"]


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "parse_jd",
            "description": "把招聘 JD 原文解析成结构化信息（公司、岗位、硬性要求、加分项、"
                           "关键词、可疑点）。分析 JD 的第一步通常调它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "jd_text": {"type": "string", "description": "JD 原文，尽量完整"},
                },
                "required": ["jd_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_experience",
            "description": "按查询词向量检索候选人的经历库，返回最相关的若干条。"
                           "需要先有已解析的 JD；query 可省略（默认用 JD 的关键词）。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索查询词，省略则用 JD 的关键词"},
                    "k": {"type": "integer", "description": "返回条数，省略则取全库"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "score_match",
            "description": "把 JD 的硬性要求逐条与候选人经历做匹配判定（matched/gap，带证据）。"
                           "需要先 parse_jd 和 search_experience。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rewrite_section",
            "description": "针对匹配结果给出带理由和夸大风险的简历改写建议。需要先 score_match。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_cover_letter",
            "description": "生成带诚实备注的投递信。需要先 score_match。",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "track_application",
            "description": "记录或更新一条投递状态。公司/岗位默认取当前已解析的 JD。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": schemas.APPLICATION_STATUSES,
                        "description": "投递状态",
                    },
                    "note": {"type": "string", "description": "备注，如投递渠道、跟进时间"},
                    "company": {"type": "string", "description": "公司名，省略则用当前 JD 的"},
                    "title": {"type": "string", "description": "岗位名，省略则用当前 JD 的"},
                },
                "required": ["status"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_applications",
            "description": "列出已记录的投递情况。用户问'我投了哪些 / 进展如何'时用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer", "description": "最多返回条数，默认 50"},
                },
            },
        },
    },
]


def _dispatch(name: str, args: dict, state: Session) -> tuple[str, bool]:
    """执行一个工具，返回 (观察结果, 是否成功)。

    观察结果是回喂给模型的文本：能摘要就摘要（省 token），完整对象留在 state 里。
    前置条件缺失时返回人话报错而不是抛异常——模型看到才能自己改正。
    """
    try:
        if name == "parse_jd":
            info = tools.parse_jd(args["jd_text"])
            state.jd_info = info
            return json.dumps(info.model_dump(), ensure_ascii=False), True

        if name == "search_experience":
            if state.jd_info is None:
                return "错误：还没有解析 JD，请先调用 parse_jd。", False
            query = args.get("query") or " ".join(state.jd_info.keywords)
            k = int(args.get("k") or tools.library_size())
            hits = tools.search_experience(query, k=k)
            state.hits = hits
            brief = "\n".join(f"{e.id} | {e.title} | {'、'.join(e.tags)}" for e in hits)
            return f"命中 {len(hits)} 条（完整内容已存入会话，下游工具可直接用）:\n{brief}", True

        if name == "score_match":
            if state.jd_info is None:
                return "错误：需要先调用 parse_jd 解析 JD。", False
            if not state.hits:
                return "错误：需要先调用 search_experience 检索经历，否则判定没有依据。", False
            match = tools.score_match(state.jd_info, state.hits, state.basics)
            state.match = match
            lines = [f"[{m.status}] {m.requirement}" for m in match.requirement_matches]
            return (f"逐条判定（{len(lines)} 条）:\n" + "\n".join(lines)
                    + f"\n总结: {match.summary}"), True

        if name == "rewrite_section":
            if state.match is None:
                return "错误：需要先调用 score_match 得到匹配结果。", False
            plan = tools.rewrite_section(state.jd_info, state.match, state.hits, state.basics)
            state.rewrite = plan
            brief = "\n".join(f"- [{s.target}] {s.reason}" for s in plan.suggestions)
            return f"给出 {len(plan.suggestions)} 条改写建议:\n{brief}", True

        if name == "generate_cover_letter":
            if state.match is None:
                return "错误：需要先调用 score_match 得到匹配结果。", False
            letter = tools.generate_cover_letter(
                state.jd_info, state.match, state.hits, state.basics)
            state.letter = letter
            notes = "\n".join(f"- {n}" for n in letter.honesty_notes)
            return f"标题: {letter.subject}\n\n{letter.body}\n\n诚实备注:\n{notes}", True

        if name == "track_application":
            company = args.get("company") or (state.jd_info.company if state.jd_info else "")
            title = args.get("title") or (state.jd_info.title if state.jd_info else "")
            if not company or not title:
                return ("错误：无法确定公司/岗位。请先调用 parse_jd，"
                        "或在参数里显式给出 company 与 title。"), False
            rec = tools.track_application(company, title, args["status"], args.get("note", ""))
            return (f"已记录：{rec['company']} · {rec['title']} · {rec['status']}"
                    f"（更新于 {rec['updated_at']}）"), True

        if name == "list_applications":
            rows = tools.list_applications(int(args.get("limit") or 50))
            if not rows:
                return "暂无投递记录。", True
            brief = "\n".join(
                f"{r['id']}. {r['company']} · {r['title']} · {r['status']}（{r['updated_at']}）"
                for r in rows)
            return f"共 {len(rows)} 条:\n{brief}", True

        return f"错误：未知工具 {name}。", False

    except Exception as e:   # 工具内部异常也变成观察结果，让模型有机会改正
        logging.warning("工具 %s 执行失败: %s", name, e)
        return f"错误：{type(e).__name__}: {e}", False


def _build_result(
    answer: str,
    stop_reason: str,
    state: Session,
    steps: list[schemas.AgentStep],
    compaction: schemas.CompactionReport,
) -> schemas.AgentResult:
    """把会话状态里的结构化产物一并带出，供界面渲染表格（决策 2026-10-07·A）。"""
    return schemas.AgentResult(
        answer=answer,
        steps=steps,
        stop_reason=stop_reason,
        prompt_tokens=state.prompt_tokens,
        completion_tokens=state.completion_tokens,
        elapsed_ms=state.elapsed_ms,
        jd_info=state.jd_info,
        hits=state.hits,
        match=state.match,
        rewrite=state.rewrite,
        letter=state.letter,
        compaction=compaction,
    )


def run_agent(
    user_input: str,
    history: list[dict] | None = None,
    max_steps: int = 6,
) -> schemas.AgentResult:
    """跑一轮 Agent：思考 → 选工具 → 执行 → 观察 → 再思考。"""
    state = Session()
    state.load_basics()

    # 历史过长先压缩，保护上下文预算（见 memory.py）
    history_messages, compaction = memory.compact_history(list(history or []))

    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history_messages)
    messages.append({"role": "user", "content": user_input})

    steps: list[schemas.AgentStep] = []
    prompt_tokens = completion_tokens = 0
    start = time.time()

    for _ in range(max_steps):
        message, usage = llm.chat_with_tools(messages, TOOL_SCHEMAS)
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]

        # 没有工具调用 = 模型认为可以回答了，循环结束
        if not message.tool_calls:
            state.prompt_tokens = prompt_tokens
            state.completion_tokens = completion_tokens
            state.elapsed_ms = int((time.time() - start) * 1000)
            return _build_result(message.content or "", "final", state, steps, compaction)

        # 把模型这轮输出（含 tool_calls）原样回填进消息历史
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [tc.model_dump() for tc in message.tool_calls],
        })

        for tc in message.tool_calls:
            step_start = time.time()
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args, observation, ok = {}, f"错误：工具参数不是合法 JSON（{e}）", False
            else:
                observation, ok = _dispatch(tc.function.name, args, state)

            if len(observation) > MAX_OBSERVATION_CHARS:
                observation = observation[:MAX_OBSERVATION_CHARS] + "\n…（已截断）"

            messages.append({"role": "tool", "tool_call_id": tc.id, "content": observation})
            steps.append(schemas.AgentStep(
                index=len(steps) + 1,
                tool=tc.function.name,
                arguments=args,
                observation=observation,
                ok=ok,
                elapsed_ms=int((time.time() - step_start) * 1000),
            ))

    state.prompt_tokens = prompt_tokens
    state.completion_tokens = completion_tokens
    state.elapsed_ms = int((time.time() - start) * 1000)
    return _build_result(
        "（已达到最大步数上限，未能给出最终答复）", "max_steps", state, steps, compaction)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        user_input = " ".join(sys.argv[1:])
    else:
        jd_file = DATA_DIR / "test_jd.txt"
        if not jd_file.exists():
            sys.exit(f"请把一份 JD 存到 {jd_file}，或把需求作为参数传入："
                     f'python src/agent.py "我想投这个岗位，帮我看看"')
        user_input = ("帮我分析这份 JD：我匹配哪些要求、缺什么，并给出简历改写建议。\n\n"
                      + jd_file.read_text(encoding="utf-8"))

    result = run_agent(user_input)

    print("\n=== 轨迹 ===")
    for s in result.steps:
        print(f"{s.index}. [{'成功' if s.ok else '失败'}] {s.tool}"
              f"({json.dumps(s.arguments, ensure_ascii=False)}) {s.elapsed_ms}ms")
    print(f"\n=== 最终答复 ===\n{result.answer}")
    print(f"\n停止原因: {result.stop_reason} | 步数: {len(result.steps)} | "
          f"tokens: {result.prompt_tokens}+{result.completion_tokens} | "
          f"总耗时: {result.elapsed_ms}ms")
