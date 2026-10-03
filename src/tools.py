"""工具集：Agent 可调用的原子能力。

每个工具 = 一个函数，输入明确、输出是已校验的 Pydantic 对象。
目前有 parse_jd（Day 2）和 search_experience（Day 3），后续几天往里加其余工具。
"""
import json
import logging
import sys
from pathlib import Path

import chromadb

import llm
import schemas
from config import CHROMA_DIR, COLLECTION_NAME, EXPERIENCE_FILE

PARSE_JD_PROMPT = """你是招聘 JD 解析器。用户会粘贴一段 JD 原文，可能有格式噪声。

你的任务：把 JD 解析成结构化信息。

解析规则：
1. must_have：硬性要求（学历、经验年限、必备技术栈），逐条拆开，不要把多条合并成一条
2. nice_to_have：加分项（原文带"优先"、"加分"字样的）
3. responsibilities：岗位职责，逐条列出
4. keywords：技术关键词，用行业通用写法（如 RAG、LangChain、FastAPI、Agent），
   不要照抄原文里的奇怪缩写   
5. red_flags：可疑点。例如：实习岗却要求 3 年经验、职责描述含糊不说具体做什么、
   薪资表述异常。没有可疑点就返回空数组，不要硬凑
6. JD 里没提到的信息返回空数组或"未知"，绝对不要编造"""


def parse_jd(jd_text: str) -> schemas.JDInfo:
    """解析 JD 原文为结构化 JDInfo。"""
    return llm.chat_structured(
        schemas.JDInfo,
        [
            {"role": "system", "content": PARSE_JD_PROMPT},
            {"role": "user", "content": jd_text},
        ],
        temperature=0.1,
    )


# ---------- search_experience（Day 3） ----------

_chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))


def _collection():
    """拿到经历集合；索引没建过就给出明确提示，而不是抛看不懂的异常。"""
    col = _chroma.get_or_create_collection(COLLECTION_NAME)
    if col.count() == 0:
        raise RuntimeError("索引为空，请先运行: python scripts/build_index.py")
    return col


def search_experience(query: str, k: int = 5) -> list[schemas.Experience]:
    """向量检索经历库，返回最相关的 k 条完整经历。

    Chroma 只存 id 和建索引文本（索引），完整对象现场回查 experience.json
    （真身），保证检索结果永远是最新数据。
    """
    vec = llm.embed_texts([query])[0]
    res = _collection().query(
        query_embeddings=[vec],
        n_results=k,
        include=["distances"],  # id 和 document 默认返回，距离用于日志观察
    )

    # 真身回查：id -> 完整经历对象
    with EXPERIENCE_FILE.open(encoding="utf-8") as f:
        by_id = {e["id"]: e for e in json.load(f)["experiences"]}

    hits = []
    for exp_id, dist in zip(res["ids"][0], res["distances"][0]):
        exp = schemas.Experience.model_validate(by_id[exp_id])
        logging.info("命中 %s (%s) 距离=%.3f", exp_id, exp.title, dist)
        hits.append(exp)
    return hits


# ---------- score_match（Day 4） ----------

SCORE_MATCH_PROMPT = """你是求职匹配判定器。给你一份 JD 的硬性要求（逐条）和候选人的相关经历，你要逐条判定匹配与否。

判定规则（必须严格遵守）：
1. status 只能是 matched 或 gap，没有第三种状态。禁止"部分匹配""基本符合"这类模糊表述
2. matched 的证据标准：必须在经历库里找到明确对应的内容，并在 evidence 里引用具体 exp_id
   和原文要点。"学习能力强""应该能快速上手""有一定基础"这类话不能作为匹配依据
3. gap 时在 gap_detail 写清具体缺什么（缺几年经验 / 缺哪个技术的实际使用）
4. gap 时若经历库里有可迁移的相关经验，如实写进 transferable 并注明来自哪条经历（exp_id）。
   可迁移经验只是补充信息，不许用它把 gap 说成 matched
5. 拿不准 matched 还是 gap 时，判 gap——漏报匹配可以人工纠正，错报匹配会误导投递决策
6. requirement_matches 的数量和顺序必须与 must_have 完全一致，一条不落
7. summary 用两三句话：最硬的地方 + 最致命的缺口，不许只说好话
8.标注『进行中』的经历只算进行中，不得计为已完成"""


def score_match(
    jd_info: schemas.JDInfo,
    experiences: list[schemas.Experience],
    basics: dict,
) -> schemas.MatchResult:
    """逐条判定 JD 硬性要求与相关经历的匹配情况。

    basics（学历/求职目标/可到岗时间等档案信息）必须传入——学历类硬性要求
    靠它判定，检索只覆盖 experiences，不传就会误判 gap。
    """
    jd_block = "\n".join(f"- {r}" for r in jd_info.must_have)
    exp_block = "\n\n".join(
        f"[{e.id}] {e.title}（{e.period}）\n标签：{'、'.join(e.tags)}\n{e.raw}\n要点：{'; '.join(e.bullets)}"
        for e in experiences
    )
    basics_block = "\n".join(f"- {k}: {v}" for k, v in basics.items() if v)
    user_content = (
        f"JD 岗位：{jd_info.title}（{jd_info.company}）\n\n"
        f"硬性要求（逐条判定）：\n{jd_block}\n\n"
        f"候选人档案：\n{basics_block}\n\n"
        f"候选人相关经历：\n{exp_block}"
    )
    return llm.chat_structured(
        schemas.MatchResult,
        [
            {"role": "system", "content": SCORE_MATCH_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )


if __name__ == "__main__":
    # 检索测试: python src/tools.py search "FastAPI 后端开发经历"
    if len(sys.argv) >= 3 and sys.argv[1] == "search":
        for e in search_experience(" ".join(sys.argv[2:])):
            print(f"{e.id} | {e.title} | {e.tags}")
        sys.exit()

    # 无参数：跑通前三环管线 parse_jd → search_experience → score_match
    jd_file = Path(__file__).parent.parent / "data" / "test_jd.txt"
    if not jd_file.exists():
        sys.exit(f"请先把一份真实 JD 保存到 {jd_file}，再运行本脚本")
    jd_text = jd_file.read_text(encoding="utf-8")

    print("=== 1. parse_jd ===")
    info = parse_jd(jd_text)
    print(f"岗位: {info.title}（{info.company}）")
    print(f"must_have: {len(info.must_have)} 条 / red_flags: {len(info.red_flags)} 条")

    print("\n=== 2. search_experience ===")
    query = " ".join(info.keywords)
    print(f"查询: {query}")
    # k=全库条数：13 条规模下全量输入判定器，漏检代价大于多几百 token；
    # 库到 50+ 条再收紧窗口并考虑 BM25 混合（决策记录 2026-10-03）
    hits = search_experience(query, k=_collection().count())
    for e in hits:
        print(f"  {e.id} | {e.title}")

    print("\n=== 3. score_match ===")
    with EXPERIENCE_FILE.open(encoding="utf-8") as f:
        basics = json.load(f)["basics"]
    match = score_match(info, hits, basics)
    for m in match.requirement_matches:
        print(f"[{m.status}] {m.requirement}")
        if m.status == "matched":
            print(f"    证据: {m.evidence}")
        else:
            print(f"    缺口: {m.gap_detail}")
            if m.transferable:
                print(f"    可迁移: {m.transferable}")
    print(f"\n总结: {match.summary}")
