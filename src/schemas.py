"""数据模型（Pydantic）。

所有 LLM 输出和接口数据都在这里定义"合同"：
模型返回的东西必须符合这里的结构，不符合就算失败。
目前有 JDInfo、Experience、MatchResult，后面用到什么再加什么。
"""
from typing import Literal

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
        description="matched 时必填：对应经历的 exp_id 和原文依据；gap 时为空字符串")
    gap_detail: str = Field(
        description="gap 时必填：具体缺什么；matched 时为空字符串")
    transferable: str = Field(
        description="gap 时如有可迁移的相关经验就写（注明来自哪条经历 exp_id），没有则为空字符串")


class MatchResult(BaseModel):
    """must_have 逐条判定的完整结果。"""
    requirement_matches: list[RequirementMatch] = Field(
        description="逐条判定，数量和顺序必须与 must_have 完全一致")
    summary: str = Field(description="两三句话：最硬的地方 + 最致命的缺口，不许只说好话")
