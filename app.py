"""求职助手 Agent —— Streamlit 演示界面。

运行: streamlit run app.py

界面分工（决策 2026-10-07·B：Tab 分栏，让一份完整分析能一屏看完）：
- 侧边栏：「＋ 新增分析」+ 历史分析列表（点开回看 / 直接删除）+ Agent 轨迹
- 主区四个 Tab：匹配报告 / 改写与投递信 / 投递记录 / 多轮对话

历史分析就是 data/runs/ 里的存档（决策 2026-10-07·A）——同一份目录既做评估追溯，
又做界面的历史记录，不另起一套存储。

界面只做展示——逻辑全在 src/ 里，这样 Day 12 的 FastAPI 能复用同一套。
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import agent  # noqa: E402
import memory  # noqa: E402
import runs  # noqa: E402
import schemas  # noqa: E402
import tools  # noqa: E402
from config import EMBED_BACKEND, LLM_MODEL  # noqa: E402

st.set_page_config(page_title="求职助手 Agent", layout="wide")

DEFAULT_INPUT = "帮我分析这份 JD：我匹配哪些要求、缺什么，并给出简历改写建议。"


def _init_state() -> None:
    st.session_state.setdefault("result", None)
    st.session_state.setdefault("current_run", None)
    st.session_state.setdefault("jd_text", "")
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("chat_log", [])


def _clear_current() -> None:
    """清空当前视图（新建分析 / 删除当前这条历史时用）。"""
    st.session_state.result = None
    st.session_state.current_run = None
    st.session_state.jd_text = ""
    st.session_state.history = []
    st.session_state.chat_log = []


# ---------- 侧边栏 ----------

def _render_sidebar_history() -> None:
    """历史分析列表。删除不二次确认——按本人要求（误删可从 git 历史找回存档）。"""
    if st.button("＋ 新增分析", key="new_analysis", type="primary", width="stretch"):
        _clear_current()
        st.rerun()

    st.markdown("#### 历史分析")
    records = runs.list_runs()
    if not records:
        st.caption("还没有分析记录。")
        return

    for r in records:
        c_open, c_del = st.columns([0.84, 0.16])
        label = f"{r['title'][:13]} · {r['timestamp'][5:]}"
        if c_open.button(label, key=f"open_{r['name']}", width="stretch"):
            try:
                loaded = runs.load_run(r["dir"])
            except ValueError as e:      # 存档坏了就提示，别让整页崩掉
                st.error(f"这条存档打不开：{e}")
            else:
                _clear_current()
                st.session_state.current_run = r["dir"]
                st.session_state.jd_text = loaded["jd_text"]
                st.session_state.result = loaded["result"]
                st.rerun()
        if c_del.button("🗑", key=f"del_{r['name']}"):
            runs.delete_run(r["dir"])
            if st.session_state.current_run == r["dir"]:
                _clear_current()
            st.rerun()


def _render_trajectory(result: schemas.AgentResult) -> None:
    """侧边栏：Agent 轨迹——面试时要讲的就是这份 steps。"""
    st.markdown("### Agent 轨迹")
    if not result.steps:
        st.caption("（这一轮没有调用工具）")
    for s in result.steps:
        icon = "✅" if s.ok else "⚠️"
        args = ", ".join(f"{k}={v}" for k, v in s.arguments.items())
        st.caption(f"{icon} {s.index}. {s.tool}({args[:40]}) · {s.elapsed_ms}ms")
        if not s.ok:
            st.warning(s.observation[:200])

    st.caption(
        f"停止原因 {result.stop_reason}｜步数 {len(result.steps)}｜"
        f"tokens {result.prompt_tokens}+{result.completion_tokens}｜"
        f"{result.elapsed_ms / 1000:.1f}s"
    )

    c = result.compaction
    if c.triggered:
        st.info(
            f"历史压缩：{c.before_tokens} → {c.after_tokens} tokens（估算）\n\n"
            f"摘要 {c.summarized_messages} 条，保留 {c.kept_messages} 条"
        )


# ---------- Tab 1 / 2 ----------

def _render_match(result: schemas.AgentResult) -> None:
    if result.match is None:
        st.info("这条记录里没有匹配结果。贴一份新 JD 点「开始分析」。")
        return

    m = result.match
    matched = [x for x in m.requirement_matches if x.status == "matched"]
    gaps = [x for x in m.requirement_matches if x.status == "gap"]

    c1, c2, c3 = st.columns(3)
    c1.metric("硬性要求", len(m.requirement_matches))
    c2.metric("匹配", len(matched))
    c3.metric("缺口", len(gaps))

    st.markdown("#### 逐条判定")
    st.dataframe(
        [
            {
                "判定": "✅ 匹配" if x.status == "matched" else "❌ 缺口",
                "要求": x.requirement,
                "依据 / 缺什么": x.evidence if x.status == "matched" else x.gap_detail,
                "可迁移经验": x.transferable or "",
            }
            for x in m.requirement_matches
        ],
        width="stretch",
        hide_index=True,
    )

    st.markdown("#### 缺口清单（投递前最该看的部分）")
    if not gaps:
        st.success("没有硬缺口。")
    for x in gaps:
        with st.expander(f"❌ {x.requirement}"):
            st.markdown(f"**缺什么**：{x.gap_detail}")
            if x.transferable:
                st.markdown(f"**可迁移**：{x.transferable}")

    st.info(m.summary)


def _render_rewrite(result: schemas.AgentResult) -> None:
    if result.rewrite is None and result.letter is None:
        st.info("这条记录里没有改写建议或投递信。")
        return

    if result.rewrite is not None:
        st.markdown("#### 简历改写建议")
        if not result.rewrite.suggestions:
            st.caption("（无改写建议）")
        for i, s in enumerate(result.rewrite.suggestions, 1):
            with st.expander(f"{i}. {s.target} — 针对：{s.requirement}"):
                st.markdown(f"**现有**：{s.original}")
                st.markdown(f"**建议**：{s.rewritten}")
                st.markdown(f"**理由**：{s.reason}")
                if s.exaggeration_risk:
                    st.warning(f"⚠️ 夸大风险：{s.exaggeration_risk}")
        st.caption(result.rewrite.summary)

    if result.letter is not None:
        st.markdown("#### 投递信")
        st.markdown(f"**标题**：{result.letter.subject}")
        st.text(result.letter.body)
        if result.letter.honesty_notes:
            st.markdown("**诚实备注（面试要能兜住）**")
            for n in result.letter.honesty_notes:
                st.caption(f"- {n}")


# ---------- Tab 3 ----------

def _render_applications() -> None:
    rows = tools.list_applications()
    st.markdown("#### 投递记录")
    if rows:
        st.dataframe(rows, width="stretch", hide_index=True)
    else:
        st.caption("还没有记录。")

    with st.form("add_application", clear_on_submit=True):
        st.markdown("**新增 / 更新一条**（同公司同岗位会更新而不是新增）")
        c1, c2, c3 = st.columns(3)
        company = c1.text_input("公司")
        title = c2.text_input("岗位")
        status = c3.selectbox("状态", schemas.APPLICATION_STATUSES)
        note = st.text_input("备注")
        if st.form_submit_button("记录"):
            if not company or not title:
                st.error("公司和岗位都要填。")
            else:
                rec = tools.track_application(company, title, status, note)
                st.success(f"已记录：{rec['company']} · {rec['title']} · {rec['status']}")
                st.rerun()


# ---------- Tab 4 ----------

def _render_chat() -> None:
    st.markdown("#### 多轮对话")
    st.caption(
        f"当前历史约 {memory.history_tokens(st.session_state.history)} tokens（估算）"
        f"｜超 {memory.DEFAULT_MAX_TOKENS} 会自动压缩"
    )

    for role, text in st.session_state.chat_log:
        with st.chat_message(role):
            st.markdown(text)

    prompt = st.chat_input("继续问，例如：把上一条缺口写进简历该怎么写？")
    if prompt:
        st.session_state.chat_log.append(("user", prompt))
        st.session_state.history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("思考中…"):
                result = agent.run_agent(prompt, history=st.session_state.history)
            st.markdown(result.answer)
        st.session_state.history.append(
            {"role": "assistant", "content": result.answer})
        st.session_state.chat_log.append(("assistant", result.answer))
        if result.compaction.triggered:
            st.info(
                f"本轮触发了历史压缩：{result.compaction.before_tokens} → "
                f"{result.compaction.after_tokens} tokens（估算）"
            )


def main() -> None:
    _init_state()

    st.title("求职助手 Agent")
    st.caption(
        f"模型 {LLM_MODEL}｜Embedding {EMBED_BACKEND}｜经历库 {tools.library_size()} 条")

    with st.sidebar:
        _render_sidebar_history()
        st.divider()
        if st.session_state.result is not None:
            _render_trajectory(st.session_state.result)
        else:
            st.caption("分析完成后，这里显示 Agent 轨迹。")

    jd_text = st.text_area(
        "粘贴招聘 JD 原文", value=st.session_state.jd_text, height=200)
    st.session_state.jd_text = jd_text

    if st.button("开始分析", type="primary"):
        if not jd_text.strip():
            st.error("先粘贴一份 JD。")
        else:
            with st.spinner("Agent 运行中…（解析 → 检索 → 判定 → 改写）"):
                result = agent.run_agent(f"{DEFAULT_INPUT}\n\n{jd_text}")
                if result.jd_info is not None:
                    run_dir = runs.save_run(jd_text, result.jd_info, result)
                    st.session_state.current_run = str(run_dir)
            st.session_state.result = result
            st.rerun()

    tab_match, tab_rewrite, tab_apps, tab_chat = st.tabs(
        ["匹配报告", "改写与投递信", "投递记录", "多轮对话"])

    empty_hint = "贴一份新 JD 点「开始分析」，或从左侧选一条历史分析。"

    with tab_match:
        if st.session_state.result is not None:
            _render_match(st.session_state.result)
        else:
            st.info(empty_hint)

    with tab_rewrite:
        if st.session_state.result is not None:
            _render_rewrite(st.session_state.result)
        else:
            st.info(empty_hint)

    with tab_apps:
        _render_applications()

    with tab_chat:
        _render_chat()


main()
