"""LLM 调用统一封装。

全项目所有大模型调用都走这个文件：
- 统一记录 token 用量和耗时（Day 12 评估"单次成本"靠它）
- 结构化输出统一走 chat_structured：校验失败自动把报错喂回去重试
"""
import json
import logging
import time

from openai import OpenAI
from pydantic import BaseModel

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    EMBED_MODEL,
    LLM_MODEL,
    ZHIPU_API_KEY,
    ZHIPU_BASE_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# 模块级单例：整个项目共用一个 client
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# embedding 用智谱（DeepSeek 没有 embedding API），同样是 OpenAI 兼容接口
embed_client = OpenAI(api_key=ZHIPU_API_KEY, base_url=ZHIPU_BASE_URL)


def chat(messages: list[dict], temperature: float = 0.7) -> str:
    """普通对话调用，返回模型回复文本。自动记录 token 与耗时。"""
    start = time.time()
    resp = client.chat.completions.create(
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


def embed_texts(texts: list[str]) -> list[list[float]]:
    """把一批文本变成向量（智谱 embedding-3），顺序与输入一一对应。"""
    start = time.time()
    resp = embed_client.embeddings.create(model=EMBED_MODEL, input=texts)
    cost_ms = int((time.time() - start) * 1000)
    # 按 index 排序再取，防止服务端返回乱序——乱序不会报错，是静默 bug
    data = sorted(resp.data, key=lambda d: d.index)
    logging.info("Embedding 调用: %s 条文本, 耗时=%sms", len(texts), cost_ms)
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
