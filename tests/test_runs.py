"""运行存档读写（runs.py）的测试。

重点覆盖：
- 存档目录结构（jd.txt / jdinfo.json / result.json）与嵌套 Pydantic 的往返
- 历史列表按时间倒序、标题取自 jdinfo
- **删除的路径守卫**——只允许删 RUNS_DIR 下的目录（rmtree 删错不可逆）
- 存档坏掉时列表不能整个崩掉
"""
import pytest

import runs
import schemas


@pytest.fixture()
def temp_runs(tmp_path, monkeypatch):
    """把 RUNS_DIR 指到临时目录，测试不碰真实存档。"""
    target = tmp_path / "runs"
    target.mkdir()
    monkeypatch.setattr(runs, "RUNS_DIR", target)
    return target


def _info(**overrides) -> schemas.JDInfo:
    data = dict(
        company="测试公司",
        title="Agent 开发实习",
        must_have=["熟悉 Python"],
        nice_to_have=[],
        responsibilities=[],
        keywords=["Python", "Agent"],
        red_flags=[],
    )
    data.update(overrides)
    return schemas.JDInfo(**data)


def test_save_and_load_roundtrip(temp_runs):
    """完整结果（含嵌套对象）能存能读，字段不丢。"""
    info = _info()
    result = schemas.AgentResult(
        answer="分析完成",
        stop_reason="final",
        jd_info=info,
        steps=[schemas.AgentStep(index=1, tool="parse_jd", arguments={"jd_text": "x"},
                                 observation="ok", ok=True, elapsed_ms=12)],
        prompt_tokens=100,
        completion_tokens=50,
    )
    run_dir = runs.save_run("JD 原文内容", info, result)

    assert (run_dir / "jd.txt").read_text(encoding="utf-8") == "JD 原文内容"
    loaded = runs.load_run(run_dir)
    assert loaded["info"].title == "Agent 开发实习"
    assert loaded["result"].answer == "分析完成"
    assert loaded["result"].steps[0].tool == "parse_jd"
    assert loaded["result"].jd_info.company == "测试公司"   # 嵌套对象也回来了


def test_save_without_result(temp_runs):
    """只解析没跑完的存档：result 为 None，不应报错。"""
    run_dir = runs.save_run("JD", _info())
    loaded = runs.load_run(run_dir)
    assert loaded["result"] is None
    assert loaded["info"].title == "Agent 开发实习"


def test_list_runs_newest_first(temp_runs):
    old = temp_runs / "2026-10-01_090000"
    old.mkdir()
    runs.save_run("旧 JD", _info(title="旧岗位"), run_dir=old)

    new = temp_runs / "2026-10-07_105900"
    new.mkdir()
    runs.save_run("新 JD", _info(title="新岗位"), run_dir=new)

    items = runs.list_runs()
    assert [i["title"] for i in items] == ["新岗位", "旧岗位"]
    assert items[0]["timestamp"] == "2026-10-07 10:59"
    assert items[0]["has_result"] is False


def test_list_tolerates_broken_jdinfo(temp_runs):
    """jdinfo.json 坏掉、但有 JD 原文：仍列出来（标题退回原文开头），不能崩。

    契约变更说明（2026-10-07）：以前是"列出来但标注不完整"，现在是
    "有可用内容就列、点得开；啥都没有就跳过"——把点不开的条目摆出来，
    正是那天 FileNotFoundError 的来源。
    """
    bad = temp_runs / "2026-10-05_120000"
    bad.mkdir()
    (bad / "jdinfo.json").write_text("{ 这不是 JSON", encoding="utf-8")
    (bad / "jd.txt").write_text("岗位职责：某段 JD 原文", encoding="utf-8")

    items = runs.list_runs()
    assert len(items) == 1
    assert items[0]["title"].startswith("岗位职责")


def test_list_falls_back_to_jd_snippet_when_title_unknown(temp_runs):
    """岗位名解析不出来时用 JD 原文开头当标题——否则历史列表会变成一列「未知」。"""
    run_dir = temp_runs / "2026-10-07_100000"
    run_dir.mkdir()
    jd = "岗位职责：参与 AI Agent 应用系统的设计与开发，包括智能客服、工具调用等场景。"
    runs.save_run(jd, _info(title="未知"), run_dir=run_dir)

    items = runs.list_runs()
    assert items[0]["title"] != "未知"
    assert items[0]["title"].startswith("岗位职责")


def test_load_recovers_info_from_result_when_jdinfo_missing(temp_runs):
    """只有 jd.txt + result.json、缺 jdinfo.json 的存档也要能打开。

    真实踩过：这种目录出现在历史列表里，点开直接 FileNotFoundError。
    数据其实一直在（result.json 里带着 jd_info），只是没去读它。
    """
    run_dir = temp_runs / "2026-10-07_111629"
    run_dir.mkdir()
    info = _info(title="从 result 里恢复的岗位")
    result = schemas.AgentResult(answer="历史回答", stop_reason="final", jd_info=info)
    # 故意不写 jdinfo.json
    (run_dir / "jd.txt").write_text("JD 原文", encoding="utf-8")
    (run_dir / "result.json").write_text(
        result.model_dump_json(indent=2), encoding="utf-8")

    loaded = runs.load_run(run_dir)
    assert loaded["info"].title == "从 result 里恢复的岗位"
    assert loaded["result"].answer == "历史回答"
    assert loaded["jd_text"] == "JD 原文"


def test_list_skips_directory_with_no_usable_content(temp_runs):
    """空目录不该出现在历史列表里——列表里的条目必须点得开。"""
    empty = temp_runs / "2026-10-07_120000"
    empty.mkdir()
    (temp_runs / "2026-10-07_090000").mkdir()
    runs.save_run("有内容", _info(), run_dir=temp_runs / "2026-10-07_090000")

    items = runs.list_runs()
    assert [i["name"] for i in items] == ["2026-10-07_090000"]


def test_load_raises_readable_error_on_empty_archive(temp_runs):
    """彻底空白的存档：报人话，不是 FileNotFoundError。"""
    empty = temp_runs / "2026-10-07_130000"
    empty.mkdir()
    with pytest.raises(ValueError) as exc:
        runs.load_run(empty)
    assert "读不出" in str(exc.value)
    assert "FileNotFoundError" not in str(exc.value)


def test_delete_run(temp_runs):
    run_dir = runs.save_run("JD", _info())
    assert runs.delete_run(run_dir) is True
    assert not run_dir.exists()
    assert runs.list_runs() == []


def test_delete_refuses_outside_runs_dir(temp_runs, tmp_path):
    """路径守卫：不许删存档目录之外的东西。"""
    outside = tmp_path / "别的地方"
    outside.mkdir()
    (outside / "重要文件.txt").write_text("不该被删", encoding="utf-8")

    with pytest.raises(ValueError):
        runs.delete_run(outside)
    assert outside.exists()
    assert (outside / "重要文件.txt").exists()


def test_delete_nonexistent_returns_false(temp_runs):
    assert runs.delete_run(temp_runs / "2026-01-01_000000") is False
