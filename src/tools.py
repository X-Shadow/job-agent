"""工具集：Agent 可调用的原子能力。

每个工具 = 一个函数，输入明确、输出是已校验的 Pydantic 对象。
现有 5 个：parse_jd / search_experience / score_match / rewrite_section / generate_cover_letter。
"""
import json
import logging
import sys
import time
from pathlib import Path

import chromadb

import llm
import schemas
from config import CHROMA_DIR, COLLECTION_NAME, DATA_DIR, EXPERIENCE_FILE, RUNS_DIR

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
    vec = llm.embed_texts([query], is_query=True)[0]
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

def _experiences_block(experiences: list[schemas.Experience]) -> str:
    """把经历列表拼成给 LLM 看的文本块（score_match / 改写 / 投递信共用）。"""
    return "\n\n".join(
        f"[{e.id}] {e.title}（{e.period}）\n标签：{'、'.join(e.tags)}\n{e.raw}\n要点：{'; '.join(e.bullets)}"
        for e in experiences
    )


def _profile_block(basics: dict) -> str:
    """把候选人档案拼成文本块。所有"要知道我是谁"的工具都必须带上它。

    纪律：漏传这个块不会报错，只会静默产出错误内容
    （已发生过 2 次：score_match 把本科判成 gap；投递信把大四写成大三）。
    新增生成/判定类工具时，先问一句"它需要档案吗"。
    """
    return "\n".join(f"- {k}: {v}" for k, v in basics.items() if v)


SCORE_MATCH_PROMPT = """你是求职匹配判定器。给你一份 JD 的硬性要求（逐条）、候选人档案和候选人的相关经历，你要逐条判定匹配与否。

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
8. 标注『进行中』的经历只算进行中，不得计为已完成"""


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
    exp_block = _experiences_block(experiences)
    basics_block = _profile_block(basics)
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


# ---------- rewrite_section / generate_cover_letter（Day 5） ----------

REWRITE_PROMPT = """你是简历改写顾问。给你 JD 的逐条匹配结果、候选人档案和候选人经历，给出改写建议。

规则：
1. 只针对真正值得改的地方给建议，按价值排序，不超过 5 条；没有值得改的就返回空列表，不要硬凑
2. rewritten 只能重新组织经历库里已有的事实（换说法、调顺序、突出重点），
   绝对禁止新增经历库里不存在的内容——没有的项目、数字、技术词一律不许出现
3. reason 写清为什么这样改、对应 JD 哪条要求
4. exaggeration_risk：改写后的说法在面试中被追问时，候选人会被问什么？哪里穿帮？
   写具体（如"会被问 Chroma 距离度量怎么选"），不要写"注意别夸大"这种废话；真没风险才留空
5. target 指明改哪里（如 "exp_001 的 bullets 第 2 条"）"""

COVER_LETTER_PROMPT = """你是投递信写手。给你 JD 信息、候选人档案、逐条匹配结果和候选人经历，写一封投递信。

规则：
1. 正文结构：为什么投这个岗 → 我有什么（只写 matched 且有证据的）→ 缺口怎么看待
   （诚实且积极，如"正在通过个人项目补"）→ 结尾
2. 只能使用经历库里存在的事实，禁止编造；不用"精通""深入""多年经验"这类与事实不符的词
3. 正文篇幅严格控制在 300-500 字（中文，不含标题与诚实备注）
4. honesty_notes：正文里哪些措辞是修辞、面试时要能兜住（如"熟悉 RAG"实际只有向量检索链路实践），
   逐条列出；一条都没有才返回空数组
5. 语气专业、不卑不亢，不写空话套话
6. 关于本人的身份与状态（学历、年级、届别、在读或已毕业、可到岗时间）只能引用候选人档案原文，
   不得自行推断或换算
7. 信是候选人第一人称自述，不得出现"经历库""exp_id""工具""匹配结果""schema"等系统内部术语，
   要换成自然表达（如"我没有相关经历"）；描述项目用自然语言，不要罗列实现细节
   （如"证据引用到具体经历"应写成"每条结论都能指到具体项目经历"）"""


def rewrite_section(
    jd_info: schemas.JDInfo,
    match: schemas.MatchResult,
    experiences: list[schemas.Experience],
    basics: dict,
) -> schemas.RewritePlan:
    """针对匹配结果给出带理由和夸大风险的简历改写建议。"""
    match_block = "\n".join(
        f"[{m.status}] {m.requirement}"
        + (f"（缺口: {m.gap_detail}）" if m.status == "gap" else f"（证据: {m.evidence}）")
        for m in match.requirement_matches
    )
    user_content = (
        f"JD 岗位：{jd_info.title}（{jd_info.company}）\n\n"
        f"候选人档案：\n{_profile_block(basics)}\n\n"
        f"逐条匹配结果：\n{match_block}\n\n"
        f"候选人经历（改写只能用这里的事实）：\n{_experiences_block(experiences)}"
    )
    return llm.chat_structured(
        schemas.RewritePlan,
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": user_content},
        ],
        temperature=0.1,
    )


def generate_cover_letter(
    jd_info: schemas.JDInfo,
    match: schemas.MatchResult,
    experiences: list[schemas.Experience],
    basics: dict,
) -> schemas.CoverLetter:
    """基于匹配结果生成投递信，附诚实备注。

    basics 必须传入：信里会写"我是谁"（学历/在读状态/可到岗时间），
    少了它模型会从证据片段里自行推断——已因此把"大四"写成过"大三"。
    """
    matched_block = "\n".join(
        f"- {m.requirement}：{m.evidence}"
        for m in match.requirement_matches if m.status == "matched"
    ) or "- （无）"
    gap_block = "\n".join(
        f"- {m.requirement}：{m.gap_detail}"
        for m in match.requirement_matches if m.status == "gap"
    ) or "- （无）"
    user_content = (
        f"JD 岗位：{jd_info.title}（{jd_info.company}）\n\n"
        f"候选人档案：\n{_profile_block(basics)}\n\n"
        f"匹配上的（信里可以理直气壮写）：\n{matched_block}\n\n"
        f"缺口（诚实处理，不回避也不硬凑）：\n{gap_block}\n\n"
        f"候选人经历（信里只能用这里的事实）：\n{_experiences_block(experiences)}"
    )
    return llm.chat_structured(
        schemas.CoverLetter,
        [
            {"role": "system", "content": COVER_LETTER_PROMPT},
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

    # 复用某次存档的解析结果: python src/tools.py reuse data/runs/2026-10-04_104312
    # parse_jd 对同一份 JD 的拆分不稳定（实测 must_have 在 4~8 条间波动，
    # temperature 降到 0 也无法消除），跨运行对比评估时必须复用同一份 JDInfo，
    # 否则测到的是解析噪声而不是判定的差异。
    if len(sys.argv) >= 2 and sys.argv[1] == "reuse":
        if len(sys.argv) < 3:
            sys.exit("用法: python src/tools.py reuse data/runs/<时间戳>")
        run_dir = Path(sys.argv[2])
        jd_text = (run_dir / "jd.txt").read_text(encoding="utf-8")
        info = schemas.JDInfo.model_validate_json(
            (run_dir / "jdinfo.json").read_text(encoding="utf-8")
        )
        print(f"=== 0. 复用存档: {run_dir} ===")
    else:
        # 无参数：跑通五步管线 parse_jd → search_experience → score_match
        #         → rewrite_section → generate_cover_letter
        jd_file = DATA_DIR / "test_jd.txt"
        if not jd_file.exists():
            sys.exit(f"请先把一份真实 JD 保存到 {jd_file}，再运行本脚本")
        jd_text = jd_file.read_text(encoding="utf-8")

        print("=== 1. parse_jd ===")
        info = parse_jd(jd_text)

        # 存档本次解析结果：JD 原文 + 结构化结果，供复跑和评估复用
        run_dir = RUNS_DIR / time.strftime("%Y-%m-%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "jd.txt").write_text(jd_text, encoding="utf-8")
        (run_dir / "jdinfo.json").write_text(
            info.model_dump_json(indent=2), encoding="utf-8"
        )
        print(f"    已存档: {run_dir}")

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

    print("\n=== 4. rewrite_section ===")
    plan = rewrite_section(info, match, hits, basics)
    if not plan.suggestions:
        print("（无改写建议）")
    for i, s in enumerate(plan.suggestions, 1):
        print(f"\n{i}. [{s.target}] 针对: {s.requirement}")
        print(f"   现有: {s.original}")
        print(f"   建议: {s.rewritten}")
        print(f"   理由: {s.reason}")
        if s.exaggeration_risk:
            print(f"   夸大风险: {s.exaggeration_risk}")
    print(f"\n小结: {plan.summary}")

    print("\n=== 5. generate_cover_letter ===")
    letter = generate_cover_letter(info, match, hits, basics)
    print(f"标题: {letter.subject}\n")
    print(letter.body)
    if letter.honesty_notes:
        print("\n诚实备注（面试要能兜住）:")
        for n in letter.honesty_notes:
            print(f"- {n}")
