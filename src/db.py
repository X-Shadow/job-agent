"""投递记录存储（SQLite + SQLAlchemy）。

这一层是纯代码、不经过 LLM——能用确定性代码做对的事就不交给模型。

表结构刻意保持简单：一条投递 = 一行。
同一「公司 + 岗位」再次记录时**更新**而不是新增——投递状态本身有生命周期
（已投递 → 已查看 → 面试中 → 已挂 / 已offer），重复插入会把同一次投递拆成多行。
"""
from datetime import datetime

from sqlalchemy import DateTime, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from config import DB_FILE


class Base(DeclarativeBase):
    pass


class Application(Base):
    """一条投递记录。"""
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    company: Mapped[str] = mapped_column(String(200))
    title: Mapped[str] = mapped_column(String(200))
    status: Mapped[str] = mapped_column(String(20))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.now, onupdate=datetime.now)


engine = create_engine(f"sqlite:///{DB_FILE}")


def init_db() -> None:
    """建表。幂等，重复调用安全。"""
    Base.metadata.create_all(engine)


def _to_dict(row: Application) -> dict:
    return {
        "id": row.id,
        "company": row.company,
        "title": row.title,
        "status": row.status,
        "note": row.note,
        "created_at": row.created_at.strftime("%Y-%m-%d %H:%M"),
        "updated_at": row.updated_at.strftime("%Y-%m-%d %H:%M"),
    }


def save_application(company: str, title: str, status: str, note: str = "") -> dict:
    """写入或更新一条投递记录，返回该记录的字典形式。"""
    init_db()
    with Session(engine) as s:
        row = s.scalar(select(Application).where(
            Application.company == company, Application.title == title))
        if row is None:
            row = Application(company=company, title=title, status=status, note=note)
            s.add(row)
        else:
            row.status = status
            if note:
                row.note = note
        s.commit()
        s.refresh(row)
        return _to_dict(row)


def list_applications(limit: int = 50) -> list[dict]:
    """按更新时间倒序列出投递记录。"""
    init_db()
    with Session(engine) as s:
        rows = s.scalars(
            select(Application).order_by(Application.updated_at.desc()).limit(limit)
        ).all()
        return [_to_dict(r) for r in rows]
