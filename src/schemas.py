"""数据模型（Pydantic）。

所有 LLM 输出和接口数据都在这里定义"合同"：
模型返回的东西必须符合这里的结构，不符合就算失败。
目前有 JDInfo、Experience、MatchResult，后面用到什么再加什么。
"""
from typing import Literal, get_args

from pydantic import BaseModel, Field


class JDInfo(BaseModel):
    """一份招聘 JD 的结构化结果。"""
    company: str = Field(description="公司名称，JD 里没有就写'未知'")
    title: str = Field(description="岗位名称")
    must_have: list[str] = Field(description="硬性要求：学历/经验/必备技术栈，逐条拆开")
    nice_to_have: list[str] = Field(description="加分项：带'优先'或'加分'字样的要求")
    responsibilities: list[str] = Field(description="岗位职责，逐条列出")
    keywords: list[str] = Field(description="技术关键词，用行业通用写法，如 RAG、FastAPI")
    red_flags: list[str] = Field(description="可疑点，如实习岗要求3年经验；没有就空列表")


class Experience(BaseModel):
    """经历库里的一条经历，与 experience.json 中的对象一一对应。"""
    id: str
    type: str          # project / competition / practice / skill / course
    title: str
    org: str
    period: str
    tags: list[str]
    raw: str
    bullets: list[str]
    metrics: dict      # 留空 {}，Day 10+ 有真实数据再填
    evidence: dict     # 证书/链接等佐证，同上


class RequirementMatch(BaseModel):
    """单条 JD 硬性要求的匹配判定。

    status 只有两态；判定线（matched/gap）和报告层次（可迁移经验）分离。
    """
    requirement: str = Field(description="JD 里的原始要求，原样保留")
    status: Literal["matched", "gap"] = Field(description="判定结果，只能是 matched 或 gap")
    evidence: str = Field(
        default="",
        description="matched 时必填：对应经历的 exp_id 和原文依据；gap 时留空")
    gap_detail: str = Field(
        description="gap 时必填：具体缺什么；matched 时为空字符串")
    transferable: str = Field(
        description="gap 时如有可迁移的相关经验就写（注明来自哪条经历 exp_id），没有则为空字符串")


class MatchResult(BaseModel):
    """must_have 逐条判定的完整结果。"""
    requirement_matches: list[RequirementMatch] = Field(
        description="逐条判定，数量和顺序必须与 must_have 完全一致")
    summary: str = Field(description="两三句话：最硬的地方 + 最致命的缺口，不许只说好话")


class RewriteSuggestion(BaseModel):
    """一条简历改写建议：只能重组经历库里已有的事实。"""
    requirement: str = Field(description="对应的 JD 要求")
    target: str = Field(description="建议改哪里，如 'exp_001 的 bullets 第 2 条'")
    original: str = Field(description="经历库里的现有写法，原样摘录")
    rewritten: str = Field(description="改写后的写法，只能重组已有事实，禁止新增")
    reason: str = Field(description="为什么这样改，对应 JD 的哪条要求")
    exaggeration_risk: str = Field(
        description="面试官追问这条时会问什么、哪里会穿帮，要具体；真没有风险才留空字符串")


class RewritePlan(BaseModel):
    """全部改写建议。"""
    suggestions: list[RewriteSuggestion] = Field(
        description="按价值排序，只给真正值得改的，不超过 5 条；没有值得改的就空列表")
    summary: str = Field(description="一句话：简历当前对这份 JD 最需要动的地方")


class CoverLetter(BaseModel):
    """投递信：正文求说服力，诚实备注列出要兜底的地方。"""
    subject: str = Field(description="邮件/消息标题")
    body: str = Field(description="信正文，可含换行，300-500 字")
    honesty_notes: list[str] = Field(
        description="正文里哪些措辞是修辞、面试时要能兜住，逐条列出；一条没有才空数组")


# ---------- 投递记录（Day 6） ----------

ApplicationStatus = Literal["已投递", "已查看", "面试中", "已挂", "已offer"]
APPLICATION_STATUSES = list(get_args(ApplicationStatus))


class ApplicationRecord(BaseModel):
    """一条投递记录的对外表示（与 db.py 的 Application 行对应）。"""
    id: int
    company: str
    title: str
    status: str
    note: str = ""
    created_at: str
    updated_at: str


# ---------- Agent 循环（Day 6） ----------

class AgentStep(BaseModel):
    """ReAct 循环里的一步：模型选择了哪个工具、传了什么参数、拿到什么观察。

    这份 trajectory 是 Day 12「工具成功率 / 平均步数 / P95 延迟」的数据源，
    所以每一步都要如实记录，包括失败的步骤。
    """
    index: int
    tool: str
    arguments: dict
    observation: str          # 工具返回（可能被截断，仅供追溯）
    ok: bool                  # 工具是否成功执行
    elapsed_ms: int


class AgentResult(BaseModel):
    """一次 Agent 运行的完整结果。"""
    answer: str
    steps: list[AgentStep] = Field(default_factory=list)
    stop_reason: str = ""     # final / max_steps / error
    prompt_tokens: int = 0
    completion_tokens: int = 0
    elapsed_ms: int = 0
