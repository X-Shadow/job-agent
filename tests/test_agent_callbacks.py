"""Agent 工具回调的测试（界面实时进度的基础）。

界面靠 on_tool_start / on_tool_end 两个回调显示每一步的进展。
这里用**假的 LLM 响应**跑完整个循环，确认回调按约定触发——不走真实 API，
所以这个测试既快又不会因为模型抽风而红。
"""
import agent
import llm


class _FakeFunction:
    def __init__(self, name: str, arguments: str) -> None:
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str) -> None:
        self.id = call_id
        self.function = _FakeFunction(name, arguments)

    def model_dump(self) -> dict:
        return {"id": self.id, "type": "function",
                "function": {"name": self.function.name,
                             "arguments": self.function.arguments}}


class _FakeMessage:
    def __init__(self, tool_calls=None, content: str = "") -> None:
        self.tool_calls = tool_calls
        self.content = content


def _fake_llm(monkeypatch, replies: list):
    """把 chat_with_tools 换成按顺序返回预设消息的假实现。"""
    queue = list(replies)
    monkeypatch.setattr(
        llm, "chat_with_tools",
        lambda messages, tools, temperature=0.1: (
            queue.pop(0),
            {"prompt_tokens": 1, "completion_tokens": 1, "elapsed_ms": 0}),
    )


def test_callbacks_fire_in_order(monkeypatch):
    """一轮工具调用 + 一轮最终答复：回调依次触发，步数正确。"""
    events: list = []

    _fake_llm(monkeypatch, [
        _FakeMessage(tool_calls=[
            _FakeToolCall("call_1", "search_experience", '{"query": "x"}')]),
        _FakeMessage(content="完成"),
    ])
    monkeypatch.setattr(
        agent, "_dispatch", lambda name, args, state: ("假观察结果", True))

    result = agent.run_agent(
        "测试输入",
        on_tool_start=lambda name, args: events.append(("start", name, args)),
        on_tool_end=lambda step: events.append(("end", step.tool, step.ok)),
    )

    assert events == [
        ("start", "search_experience", {"query": "x"}),
        ("end", "search_experience", True),
    ]
    assert result.answer == "完成"
    assert result.stop_reason == "final"
    assert len(result.steps) == 1
    assert result.steps[0].observation == "假观察结果"


def test_on_tool_end_records_failure(monkeypatch):
    """工具失败时 ok=False 要如实传出去——界面靠它显示 ⚠️。"""
    seen: list = []

    _fake_llm(monkeypatch, [
        _FakeMessage(tool_calls=[_FakeToolCall("call_1", "score_match", "{}")]),
        _FakeMessage(content="知道了"),
    ])
    monkeypatch.setattr(
        agent, "_dispatch", lambda name, args, state: ("错误：缺少前置步骤", False))

    result = agent.run_agent("测试", on_tool_end=lambda step: seen.append(step.ok))

    assert seen == [False]
    assert result.steps[0].ok is False


def test_works_without_callbacks(monkeypatch):
    """不传回调 = 命令行行为，必须照常工作。"""
    _fake_llm(monkeypatch, [_FakeMessage(content="直接回答")])

    result = agent.run_agent("测试")

    assert result.answer == "直接回答"
    assert result.steps == []
    assert result.stop_reason == "final"
