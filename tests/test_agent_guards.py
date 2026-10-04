"""Agent 前置条件护栏的测试。

方案 A（会话状态）的代价是**隐式顺序依赖**——模型可能没解析 JD 就调 score_match。
所以护栏必须给出人话报错，让模型读到之后能自己补齐前置步骤。
这些用例就是在验证"报错是否可读、是否指向了正确的下一步"。

出处：原为 2026-10-04 的临时单测，Day 8 升级为 pytest 用例。
不走 LLM，全部是纯逻辑校验。
"""
import agent


def _empty_state() -> agent.Session:
    """一个刚开工的会话：有档案，但还没解析 JD、没检索、没判定。"""
    state = agent.Session()
    state.load_basics()
    return state


def test_search_requires_parsed_jd():
    obs, ok = agent._dispatch("search_experience", {}, _empty_state())
    assert ok is False
    assert "parse_jd" in obs          # 报错要指向正确的下一步


def test_score_match_requires_parsed_jd():
    obs, ok = agent._dispatch("score_match", {}, _empty_state())
    assert ok is False
    assert "parse_jd" in obs


def test_rewrite_requires_match():
    obs, ok = agent._dispatch("rewrite_section", {}, _empty_state())
    assert ok is False
    assert "score_match" in obs


def test_cover_letter_requires_match():
    obs, ok = agent._dispatch("generate_cover_letter", {}, _empty_state())
    assert ok is False
    assert "score_match" in obs


def test_track_requires_company_and_title():
    """没有已解析的 JD、也没显式给公司岗位时，应拒绝。"""
    obs, ok = agent._dispatch("track_application", {"status": "已投递"}, _empty_state())
    assert ok is False
    assert "公司" in obs


def test_unknown_tool_rejected():
    obs, ok = agent._dispatch("不存在的工具", {}, _empty_state())
    assert ok is False
    assert "未知工具" in obs


def test_track_accepts_explicit_company_and_title(temp_db):
    """旁路：显式给出公司/岗位时，不依赖 JD 也能记录。"""
    obs, ok = agent._dispatch(
        "track_application",
        {"status": "已投递", "company": "显式公司", "title": "显式岗位"},
        _empty_state(),
    )
    assert ok is True
    assert "已记录" in obs
