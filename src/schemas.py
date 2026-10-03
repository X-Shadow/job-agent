"""数据模型（Pydantic）。

所有 LLM 输出和接口数据都在这里定义"合同"：
模型返回的东西必须符合这里的结构，不符合就算失败。
今天只定义 JDInfo，后面几天用到什么再加什么。
"""
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
