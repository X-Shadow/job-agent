"""把 HF 缓存里的本地 embedding 模型暂存到 models/，供 Docker 构建时 COPY 进镜像。

为什么需要这一步：
模型默认缓存在 ~/.cache/huggingface/，那个路径**在 Docker 构建上下文之外**——
`COPY` 够不着上下文以外的文件。所以先把它暂存到项目内，构建时才拿得到。

models/ 已加进 .gitignore：模型不该进 git（92MB，且能从 HF 重新获取）。

跑法：python scripts/stage_model.py
"""
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from config import LOCAL_EMBED_MODEL  # noqa: E402

HUB_CACHE = Path.home() / ".cache" / "huggingface" / "hub"
STAGE_DIR = Path(__file__).resolve().parent.parent / "models" / "hub"


def cached_dir_name(model_id: str) -> str:
    """HF 缓存目录命名规则：BAAI/bge-small-zh-v1.5 → models--BAAI--bge-small-zh-v1.5"""
    return "models--" + model_id.replace("/", "--")


def main() -> None:
    src = HUB_CACHE / cached_dir_name(LOCAL_EMBED_MODEL)
    if not src.is_dir():
        sys.exit(f"缓存里找不到模型：{src}\n"
                 f"先跑一次 python scripts/build_index.py，让它把模型载入缓存")

    dst = STAGE_DIR / src.name
    if dst.exists():
        shutil.rmtree(dst)          # 全量重来，避免暂存目录里混着旧文件
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)

    size_mb = sum(f.stat().st_size for f in dst.rglob("*") if f.is_file()) / 1024 / 1024
    print(f"已暂存模型 → {dst}")
    print(f"体积：{size_mb:.0f} MB")
    print("提示：models/ 在 .gitignore 中，不会进 git")


if __name__ == "__main__":
    main()
