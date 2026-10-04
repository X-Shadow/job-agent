"""建向量索引：experience.json → Chroma（全量重建）。

运行: python scripts/build_index.py

每次都全量重建（删旧集合再建），13 条数据几秒完成，简单可靠。
经历增删改之后重跑一次即可让索引和真身保持一致。
"""
import json
import sys
from pathlib import Path

# 本脚本在 scripts/ 下运行，sys.path[0] 是 scripts/，
# 手动把 src/ 加进搜索路径才能 import config / llm
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chromadb  # noqa: E402

import llm  # noqa: E402
from config import (  # noqa: E402
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBED_BACKEND,
    EMBED_MODEL,
    EXPERIENCE_FILE,
    LOCAL_EMBED_MODEL,
)


def assemble_text(exp: dict) -> str:
    """拼接一条经历的索引文本：title 放最前，tags 单独一行。

    只拼 title/raw/bullets/tags——它们决定"能被什么词搜到"。
    metrics/evidence 不进：无语义贡献，只会稀释向量。
    """
    parts = [exp["title"], exp["raw"], *exp["bullets"], " ".join(exp["tags"])]
    return "\n".join(parts)


def main() -> None:
    with EXPERIENCE_FILE.open(encoding="utf-8") as f:
        experiences = json.load(f)["experiences"]

    ids = [e["id"] for e in experiences]
    texts = [assemble_text(e) for e in experiences]
    print(f"[1/3] 已读入 {len(ids)} 条经历")

    vectors = llm.embed_texts(texts)
    model_name = LOCAL_EMBED_MODEL if EMBED_BACKEND == "local" else EMBED_MODEL
    print(f"[2/3] 已生成 {len(vectors)} 条向量 (后端: {EMBED_BACKEND}, 模型: {model_name}, 维度: {len(vectors[0])})")

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass  # 第一次运行还没有这个集合，跳过即可
    col = client.get_or_create_collection(COLLECTION_NAME)
    col.add(ids=ids, documents=texts, embeddings=vectors)
    print(f"[3/3] 已写入 Chroma: {CHROMA_DIR}")
    print(f"完成。集合 {COLLECTION_NAME!r} 共 {col.count()} 条")


if __name__ == "__main__":
    main()
