"""LLM 调用统一封装。

全项目所有大模型调用都走这个文件：
- 统一记录 token 用量和耗时（Day 12 评估"单次成本"靠它）
- 结构化输出统一走 chat_structured：校验失败自动把报错喂回去重试
"""
import json
import logging
import time
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    EMBED_BACKEND,
    EMBED_MODEL,
    LLM_MODEL,
    LOCAL_EMBED_MODEL,
    QUERY_INSTRUCTION,
    USE_QUERY_INSTRUCTION,
    ZHIPU_API_KEY,
    ZHIPU_BASE_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 两个 client 都**惰性构造**，不在 import 时创建。
# 原因：Docker 构建期只跑本地 embedding（build_index.py 会 import 本模块），
# 那时没有、也不需要对话密钥——在 import 阶段就要求密钥，会让"不需要密钥的活儿"也做不了。
# 顺带好处：缺密钥时报的是明确的中文提示，而不是 openai SDK 的英文异常。
_client = None
_embed_client = None


def _get_client():
    """对话模型 client（DeepSeek）。首次真正用到时才创建。"""
    global _client
    if _client is None:
        if not DEEPSEEK_API_KEY:
            raise RuntimeError(
                "缺少 DEEPSEEK_API_KEY：在 .env 里填好，或运行时用环境变量传入")
        _client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)
    return _client


def _get_embed_client():
    """embedding client（智谱）。只有 EMBED_BACKEND=zhipu 时才会被创建。"""
    global _embed_client
    if _embed_client is None:
        if not ZHIPU_API_KEY:
            raise RuntimeError("缺少 ZHIPU_API_KEY：EMBED_BACKEND=zhipu 时必须提供")
        _embed_client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)
    return _embed_client


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    """普通对话调用，返回模型回复文本。自动记录 token 与耗时。"""
    start = time.time()
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    cost_ms = int((time.time() - start) * 1000)
    usage = resp.usage
    logging.info(
        "LLM 调用: prompt=%s completion=%s tokens, 耗时=%sms",
        usage.prompt_tokens, usage.completion_tokens, cost_ms,
    )
    return resp.choices[0].message.content


def chat_with_tools(
    messages: list[dict],
    tools: list[dict],
    temperature: float = 0.1,
) -> tuple[Any, dict]:
    """带工具定义的对话调用（原生 function calling）。

    返回 (message, usage)。message 可能带 tool_calls——有就说明模型要调工具，
    没有就是最终答复，由调用方（agent.py 的循环）判断。usage 供累计 token 与耗时。
    """
    start = time.time()
    resp = _get_client().chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        tools=tools,
        temperature=temperature,
    )
    cost_ms = int((time.time() - start) * 1000)
    usage = resp.usage
    choice = resp.choices[0]
    logging.info(
        "LLM(工具) 调用: prompt=%s completion=%s tokens, 耗时=%sms, finish=%s",
        usage.prompt_tokens, usage.completion_tokens, cost_ms, choice.finish_reason,
    )
    return choice.message, {
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "elapsed_ms": cost_ms,
    }


def chat_structured(
    schema: type[BaseModel],
    messages: list[dict],
    temperature: float = 0.1,
    retries: int = 2,
) -> BaseModel:
    """结构化输出：要求模型按 schema 返回 JSON，校验失败自动重试。

    重试时把报错信息喂回模型让它自己修正——最简单有效的错误恢复。
    返回值是已校验的 Pydantic 实例，字段类型有保证。
    """
    schema_str = json.dumps(schema.model_json_schema(), ensure_ascii=False, indent=2)
    rule = (
        "输出要求：只输出一个 JSON 对象，禁止任何解释文字或 markdown 标记。\n"
        f"JSON 必须符合如下 JSON Schema：\n{schema_str}"
    )
    work = list(messages)
    if work and work[0].get("role") == "system":
        # 已有 system 就把格式要求追加进去，避免出现两条 system 消息
        work[0] = {**work[0], "content": work[0]["content"] + "\n\n" + rule}
    else:
        work.insert(0, {"role": "system", "content": rule})

    last_err: Exception | None = None
    for attempt in range(1 + retries):
        text = chat(work, temperature=temperature)
        try:
            data = _extract_json(text)
            return schema.model_validate(data)
        except Exception as e:
            last_err = e
            logging.warning("第 %s 次输出校验失败: %s", attempt + 1, e)
            work.append({"role": "assistant", "content": text})
            work.append({
                "role": "user",
                "content": f"你上一条输出没通过校验：{e}\n请重新只输出符合要求的 JSON。",
            })

    raise RuntimeError(f"结构化输出在 {1 + retries} 次尝试后仍失败: {last_err}")


_local_model = None


def _get_local_model():
    """懒加载本地 embedding 模型。

    只在真正调用时才 import torch / 载模型——否则每次跑脚本都要白等几秒。
    模型已在 HF 缓存里，不需要联网下载。
    """
    global _local_model
    if _local_model is None:
        from sentence_transformers import SentenceTransformer

        logging.info("加载本地 embedding 模型 %s（首次较慢）...", LOCAL_EMBED_MODEL)
        _local_model = SentenceTransformer(LOCAL_EMBED_MODEL)
    return _local_model


def embed_texts(texts: list[str], is_query: bool = False) -> list[list[float]]:
    """把一批文本变成向量，顺序与输入一一对应。按配置分派到本地或智谱后端。

    is_query=True 表示这是检索查询（可以加 bge 的查询指令）；False 表示文档侧。
    """
    if EMBED_BACKEND == "local":
        return _embed_local(texts, is_query)
    return _embed_zhipu(texts)


def _embed_local(texts: list[str], is_query: bool) -> list[list[float]]:
    """本地 bge-small-zh-v1.5。池化与归一化由模型自带模块决定（CLS + Normalize）。"""
    payload = list(texts)
    if is_query and USE_QUERY_INSTRUCTION:
        payload = [QUERY_INSTRUCTION + t for t in payload]
    start = time.time()
    vecs = _get_local_model().encode(payload, show_progress_bar=False)
    cost_ms = int((time.time() - start) * 1000)
    logging.info(
        "Embedding(本地): %s 条文本, 维度=%s, 耗时=%sms",
        len(texts), len(vecs[0]), cost_ms,
    )
    return [v.tolist() for v in vecs]


def _embed_zhipu(texts: list[str]) -> list[list[float]]:
    """智谱 embedding-3（OpenAI 兼容接口）。"""
    start = time.time()
    resp = _get_embed_client().embeddings.create(model=EMBED_MODEL, input=texts)
    cost_ms = int((time.time() - start) * 1000)
    # 按 index 排序再取，防止服务端返回乱序——乱序不会报错，是静默 bug
    data = sorted(resp.data, key=lambda d: d.index)
    logging.info("Embedding(智谱): %s 条文本, 耗时=%sms", len(texts), cost_ms)
    return [d.embedding for d in data]


def _extract_json(text: str) -> dict:
    """从模型回复里抠出 JSON 对象（容忍 markdown 代码块和前后废话）。"""
    text = text.strip()
    if text.startswith("```"):
        # 剥掉 ```json ... ``` 包裹
        text = text.split("\n", 1)[1] if "\n" in text else text
        end = text.rfind("```")
        if end != -1:
            text = text[:end]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"回复里找不到 JSON 对象: {text[:100]}")
    return json.loads(text[start:end + 1])
