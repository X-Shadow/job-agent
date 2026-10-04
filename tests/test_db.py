"""投递记录（纯代码层）的测试。

出处：原为 2026-10-04 的临时单测，Day 8 升级为 pytest 用例。
这一层不用 LLM——"该用代码的地方不用模型"，测试也就特别踏实。
"""
import pytest

import tools


def test_save_and_update_same_record(temp_db):
    """同一「公司+岗位」再次记录时更新，而不是新增重复行。"""
    first = tools.track_application("测试公司A", "Agent 开发实习", "已投递", "BOSS 上投的")
    assert first["status"] == "已投递"

    second = tools.track_application("测试公司A", "Agent 开发实习", "面试中", "约了下周一")
    assert second["id"] == first["id"]        # 同一行
    assert second["status"] == "面试中"        # 状态被更新
    assert second["note"] == "约了下周一"
    assert len(tools.list_applications()) == 1  # 没有多出一行


def test_note_not_overwritten_when_empty(temp_db):
    """传空备注时不应把已有备注清掉。"""
    tools.track_application("公司A", "岗位1", "已投递", "内推")
    row = tools.track_application("公司A", "岗位1", "已查看")   # 不传 note
    assert row["note"] == "内推"


def test_list_orders_by_updated_desc(temp_db):
    """列表按更新时间倒序——后写入的排前面。"""
    tools.track_application("公司A", "岗位1", "已投递")
    tools.track_application("公司B", "岗位2", "已投递")
    rows = tools.list_applications()
    assert len(rows) == 2
    assert rows[0]["company"] == "公司B"


def test_invalid_status_rejected(temp_db):
    """状态不在枚举内时报错，且报错里列出合法取值。"""
    with pytest.raises(ValueError) as exc:
        tools.track_application("公司C", "岗位", "已通过")
    message = str(exc.value)
    assert "已投递" in message and "已offer" in message
