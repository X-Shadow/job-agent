"""全局配置：路径、密钥、模型名。

所有模块都从这里取配置，不要在别处硬编码 key 或路径。
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Windows 控制台默认 GBK 编码，脚本里 print 中文会报 UnicodeEncodeError。
# 这一行把它改成 UTF-8，修掉这个坑。
sys.stdout.reconfigure(encoding="utf-8")

# 读取 .env 里的密钥（.env 已在 .gitignore 中，不会被提交）
load_dotenv()

# 本地 embedding 模型已完整在 HF 缓存里。强制离线模式，否则 huggingface_hub
# 会对每个文件发 HEAD 请求检查更新，网络不通时重试 5 次指数退避——
# 实测一次 embedding 从 1 秒变成 178 秒。离线模式只读缓存，不联网。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

EXPERIENCE_FILE = DATA_DIR / "experience.json"   # 经历库（核心数据）
CHROMA_DIR = DATA_DIR / "chroma"                 # 向量库持久化目录
DB_FILE = DATA_DIR / "applications.db"           # 投递记录
COLLECTION_NAME = "experience"                   # Chroma 集合名（建索引/检索共用）
RUNS_DIR = DATA_DIR / "runs"                     # 每次运行存档（JD 原文 + 解析结果）

# ---------- 密钥 ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")   # 对话模型
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")         # embedding

# ---------- 模型 ----------
LLM_MODEL = "deepseek-flash"   # DeepSeek 当前可用: deepseek-flash / deepseek-v4-pro
EMBED_MODEL = "embedding-3"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"

# ---------- Embedding 后端 ----------
# "local"：本地 bge-small-zh-v1.5（免费、离线、512 维，模型已在 HF 缓存里）
# "zhipu"：智谱 embedding-3 API（2048 维，需要余额）
# 切换后端后必须重建索引：python scripts/build_index.py
EMBED_BACKEND = "local"
LOCAL_EMBED_MODEL = "BAAI/bge-small-zh-v1.5"

# bge 查询指令：只作用于查询，文档侧不加。官方说 v1.5 加不加差异很小，
# 哪个更好要在自己任务上测——所以留成开关做 A/B。
USE_QUERY_INSTRUCTION = False
QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def check() -> None:
    """自检：密钥是否配好、数据文件是否存在。写完 .env 后跑一次确认。"""
    print(f"项目根目录: {ROOT}")
    print(f"Embedding:    {EMBED_BACKEND}"
          + (f"（{LOCAL_EMBED_MODEL}）" if EMBED_BACKEND == "local"
             else f"（{EMBED_MODEL}）"))
    print(f"DeepSeek Key: {'已配置' if DEEPSEEK_API_KEY else '缺失 <- 去 .env 里填'}")

    # 智谱 Key 只在用智谱 embedding 后端时才需要，别的后端下缺失不算问题
    zhipu_state = "已配置" if ZHIPU_API_KEY else "未配置"
    note = "" if EMBED_BACKEND == "zhipu" else f"（当前 EMBED_BACKEND={EMBED_BACKEND}，不需要）"
    print(f"智谱 Key:     {zhipu_state}{note}")

    print(f"经历库文件:   {'存在' if EXPERIENCE_FILE.exists() else '缺失'}")


if __name__ == "__main__":
    check()
