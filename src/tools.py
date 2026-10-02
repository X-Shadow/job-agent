"""工具集：Agent 可调用的原子能力。

每个工具 = 一个函数，输入明确、输出是已校验的 Pydantic 对象。
今天只写 parse_jd，Day 3-5 会往里加其余工具。
"""
import json
import sys
from pathlib import Path

import llm
import schemas

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


if __name__ == "__main__":
    jd_file = Path(__file__).parent.parent / "data" / "test_jd.txt"
    if not jd_file.exists():
        sys.exit(f"请先把一份真实 JD 保存到 {jd_file}，再运行本脚本")
    jd_text = jd_file.read_text(encoding="utf-8")

    info = parse_jd(jd_text)
    output = json.dumps(info.model_dump(), ensure_ascii=False, indent=2)
    print(output)

    # 同时存档一份（UTF-8），方便 A/B 对比。文件名跟在命令后面：
    #   python src/tools.py output_noRule4.json
    out_name = sys.argv[1] if len(sys.argv) > 1 else "output_latest.json"
    out_file = Path(__file__).parent.parent / "data" / out_name
    out_file.write_text(output, encoding="utf-8")
    print(f"\n[已存档] {out_file}")
