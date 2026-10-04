"""pytest 公共配置。

两件事：
1. 把 src/ 加进导入路径——测试里才能 `import db / tools / agent`
2. 提供 temp_db fixture：把 db.engine 换成临时库，
   测试**永远不碰真实的 data/applications.db**
"""
import sys
from pathlib import Path

import pytest
from sqlalchemy import create_engine

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


@pytest.fixture()
def temp_db(tmp_path, monkeypatch):
    """用临时 SQLite 替换真实库（monkeypatch 会在用例结束后自动还原）。"""
    import db

    engine = create_engine(f"sqlite:///{tmp_path / 'test.db'}")
    monkeypatch.setattr(db, "engine", engine)
    db.Base.metadata.create_all(engine)
    yield
