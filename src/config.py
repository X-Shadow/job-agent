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

# ---------- 路径 ----------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"

EXPERIENCE_FILE = DATA_DIR / "experience.json"   # 经历库（核心数据）
CHROMA_DIR = DATA_DIR / "chroma"                 # 向量库持久化目录
DB_FILE = DATA_DIR / "applications.db"           # 投递记录

# ---------- 密钥 ----------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")   # 对话模型
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")         # embedding

# ---------- 模型 ----------
LLM_MODEL = "deepseek-chat"
EMBED_MODEL = "embedding-3"

DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
ZHIPU_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"


def check() -> None:
    """自检：密钥是否配好、数据文件是否存在。写完 .env 后跑一次确认。"""
    print(f"项目根目录: {ROOT}")
    print(f"DeepSeek Key: {'已配置' if DEEPSEEK_API_KEY else '缺失 <- 去 .env 里填'}")
    print(f"智谱 Key:     {'已配置' if ZHIPU_API_KEY else '缺失 <- 去 .env 里填'}")
    print(f"经历库文件:   {'存在' if EXPERIENCE_FILE.exists() else '缺失'}")


if __name__ == "__main__":
    check()
