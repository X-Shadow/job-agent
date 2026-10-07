"""运行存档的读写：每次分析存一个目录。

目录结构：
    data/runs/<YYYY-MM-DD_HHMMSS>/
        jd.txt        JD 原文
        jdinfo.json   parse_jd 的结构化结果（供复用，见 tools.py 的 reuse 分支）
        result.json   完整 AgentResult（匹配 / 改写 / 投递信 / 轨迹）——分析跑完才有

同一份存档同时服务两件事，所以不再另起一套存储：
1. **评估可追溯**（Day 4 决策：parse_jd 输出不稳定，跨运行对比必须复用同一份 JDInfo）
2. **界面的「历史分析」**（点开回看当时那份 JD 的完整判定）
"""
import logging
import shutil
import time
from pathlib import Path

import schemas
from config import RUNS_DIR


def new_run_dir() -> Path:
    """按当前时间戳建一个存档目录。"""
    run_dir = RUNS_DIR / time.strftime("%Y-%m-%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_run(
    jd_text: str,
    info: schemas.JDInfo,
    result: schemas.AgentResult | None = None,
    run_dir: Path | None = None,
) -> Path:
    """写一次分析：JD 原文 + 解析结果（跑完的话再写完整结果）。返回存档目录。"""
    target = run_dir or new_run_dir()
    (target / "jd.txt").write_text(jd_text, encoding="utf-8")
    (target / "jdinfo.json").write_text(
        info.model_dump_json(indent=2), encoding="utf-8")
    if result is not None:
        (target / "result.json").write_text(
            result.model_dump_json(indent=2), encoding="utf-8")
    return target


def _format_stamp(name: str) -> str:
    """目录名 2026-10-07_105612 → 2026-10-07 10:56。"""
    date, _, clock = name.partition("_")
    return f"{date} {clock[:2]}:{clock[2:4]}" if len(clock) >= 4 else name


def _read_result(d: Path) -> schemas.AgentResult | None:
    """读 result.json；不存在或读不出都返回 None。"""
    result_file = d / "result.json"
    if not result_file.exists():
        return None
    try:
        return schemas.AgentResult.model_validate_json(
            result_file.read_text(encoding="utf-8"))
    except Exception as e:
        logging.warning("存档 %s 的 result.json 读不出: %s", d.name, e)
        return None


def _read_info(d: Path) -> schemas.JDInfo | None:
    """恢复 JDInfo：优先 jdinfo.json，退而用 result.json 里带的 jd_info。

    为什么要兜底：存档可能只写了 result.json（例如运行中途被打断）。
    result.json 里本来就带着 jd_info，数据在，只是不在那个文件里——
    读不到就说"读不到"，而不是让调用方炸掉。
    """
    info_file = d / "jdinfo.json"
    if info_file.exists():
        try:
            return schemas.JDInfo.model_validate_json(
                info_file.read_text(encoding="utf-8"))
        except Exception as e:
            logging.warning("存档 %s 的 jdinfo.json 读不出: %s", d.name, e)

    result = _read_result(d)
    if result is not None and result.jd_info is not None:
        logging.info("存档 %s 缺 jdinfo.json，已从 result.json 恢复", d.name)
        return result.jd_info
    return None


def list_runs(limit: int = 50) -> list[dict]:
    """列出历史分析，按时间倒序（目录名带时间戳，直接倒排即可）。"""
    if not RUNS_DIR.exists():
        return []

    items = []
    dirs = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
    for d in dirs:
        if len(items) >= limit:
            break

        jd_file = d / "jd.txt"
        jd_text = jd_file.read_text(encoding="utf-8") if jd_file.exists() else ""
        info = _read_info(d)

        # 既没有解析结果、又没有 JD 原文的目录不算一条记录，跳过。
        # 原则：**列表里出现的条目必须点得开**，否则就是让用户去踩异常。
        if info is None and not jd_text.strip():
            logging.warning("跳过无内容的存档目录: %s", d.name)
            continue

        title = info.title if info else ""
        company = info.company if info else ""
        # 岗位名解析不出来时（JD 原文常没有单独的岗位行）退回用原文开头，
        # 否则历史列表会变成一列「未知」，等于没有历史
        if title in ("", "未知"):
            title = " ".join(jd_text.split())[:16] or "（无标题）"

        items.append({
            "dir": str(d),
            "name": d.name,
            "timestamp": _format_stamp(d.name),
            "title": title,
            "company": company,
            "has_result": (d / "result.json").exists(),
        })
    return items


def load_run(run_dir: str | Path) -> dict:
    """读回一次分析：{jd_text, info, result}。

    容错：`jdinfo.json` 缺失时从 `result.json` 里的 jd_info 恢复；`jd.txt` 缺失时给空串。
    两者都拿不到才报错——报的也是人话，而不是 FileNotFoundError。
    """
    d = Path(run_dir)
    if not d.is_dir():
        raise ValueError(f"存档目录不存在：{d}")

    info = _read_info(d)
    if info is None:
        raise ValueError(
            f"存档不完整、读不出 JD 内容：{d.name}"
            f"（缺 jdinfo.json，result.json 里也没有 jd_info）")

    jd_file = d / "jd.txt"
    return {
        "jd_text": jd_file.read_text(encoding="utf-8") if jd_file.exists() else "",
        "info": info,
        "result": _read_result(d),
    }


def delete_run(run_dir: str | Path) -> bool:
    """删除一次存档。

    只允许删 RUNS_DIR 底下的目录——防止传进来一个别的路径（比如 data/experience.json
    的父目录）被 shutil.rmtree 一把端掉。这条检查不是多余的谨慎：
    rmtree 删错东西是不可逆的。
    """
    target = Path(run_dir).resolve()
    root = RUNS_DIR.resolve()
    if target.parent != root:
        raise ValueError(f"拒绝删除 {target}：不在存档目录 {root} 下")
    if not target.is_dir():
        return False
    shutil.rmtree(target)
    logging.info("已删除存档 %s", target.name)
    return True


def latest_run_dir() -> Path | None:
    """最近一次存档目录（实验脚本用，避免写死时间戳）。"""
    if not RUNS_DIR.exists():
        return None
    dirs = sorted((p for p in RUNS_DIR.iterdir() if p.is_dir()), reverse=True)
    return dirs[0] if dirs else None
