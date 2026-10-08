# 求职助手 Agent —— 运行镜像
#
# 构建前必须先暂存本地 embedding 模型（它在 Docker 构建上下文之外，COPY 够不着）：
#     python scripts/stage_model.py
# 然后：
#     docker build -t job-agent .
#     docker run -p 8501:8501 -e DEEPSEEK_API_KEY=你的key job-agent

FROM python:3.14-slim

# HF_HOME 指向烤进镜像的模型目录；HF_HUB_OFFLINE 强制离线——
# 否则 huggingface_hub 会对每个文件发 HEAD 请求检查更新，
# 网络不通时重试 5 次指数退避（实测一次 embedding 从 1 秒变 178 秒）
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1

WORKDIR /app

# 包索引可在构建时覆盖——国内服务器直连官方源常常很慢或不通。
# 服务器上构建时用（这两个国内源已实测可用）：
#   --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
#   --build-arg TORCH_INDEX=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu
# 注意：mirrors.aliyun.com/pytorch-wheels/cpu 虽然能打开，但**不是合法的 pip 索引**
#      （pip index versions 报 No matching distribution），别用那个。
ARG PIP_INDEX=https://pypi.org/simple
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# 1) torch 单独先装 CPU 版。
#    Linux 上 pip install torch 默认拉 CUDA 版（带一堆 nvidia-* 依赖，2GB+），
#    我们只做 CPU 推理，从 CPU 索引装能省掉一半以上体积。
#    --retries/--timeout 调大：实测容器到 download.pytorch.org 的连接会偶发
#    SSL EOF（宿主机同一条网络没事），pip 默认 5 次重试用尽就整个 build 失败。
RUN pip install --retries 10 --timeout 60 \
        --index-url ${TORCH_INDEX} torch

# 2) 其余依赖。requirements.txt 里钉了 torch==...，已经装过了就过滤掉，
#    避免 pip 又去拉一份 CUDA 版覆盖掉。
COPY requirements.txt .
RUN grep -v '^torch==' requirements.txt > /tmp/req.txt \
 && pip install --retries 10 --timeout 60 -i ${PIP_INDEX} -r /tmp/req.txt

# 3) 本地 embedding 模型（约 92MB，构建前用 stage_model.py 暂存好）
COPY models/ /models/

# 4) 应用代码（个人数据与索引在 .dockerignore 里排除了）
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY app.py ./
COPY data/experience.json ./data/experience.json

# 5) 构建期生成向量索引。本地 embedding 不需要任何密钥，所以构建期能跑。
#    代价：改了 data/experience.json 就要重新 build 镜像。
RUN python scripts/build_index.py

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# 必须监听 0.0.0.0，否则容器外访问不到
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
