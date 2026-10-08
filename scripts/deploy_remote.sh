#!/usr/bin/env bash
# 在云服务器上执行：构建镜像 + 启动容器 + 打印访问地址
#
# 用法（在项目根目录）：
#     bash scripts/deploy_remote.sh
#
# 详细步骤与排错见 docs/deploy.md
set -euo pipefail

IMAGE=job-agent
CONTAINER=job-agent
PORT=8501

# 国内镜像源（已实测可用）。想用官方源就覆盖这两个环境变量。
PIP_INDEX="${PIP_INDEX:-https://pypi.tuna.tsinghua.edu.cn/simple}"
TORCH_INDEX="${TORCH_INDEX:-https://mirror.sjtu.edu.cn/pytorch-wheels/cpu}"

cd "$(dirname "$0")/.."

echo "==> 检查前置条件"
command -v docker >/dev/null 2>&1 \
  || { echo "没装 Docker。先跑：curl -fsSL https://get.docker.com | sh"; exit 1; }
[ -f .env ] \
  || { echo "缺少 .env（API 密钥）。先配好再跑。"; exit 1; }
[ -d models/hub ] \
  || { echo "缺少 models/（本地 embedding 模型，不在 git 里）。"
       echo "在你自己电脑上跑 python scripts/stage_model.py，再 scp -r models 到服务器。"
       exit 1; }

echo "==> 构建镜像（torch 约 942MB，第一次会慢）"
docker build -t "$IMAGE" \
  --build-arg PIP_INDEX="$PIP_INDEX" \
  --build-arg TORCH_INDEX="$TORCH_INDEX" .

echo "==> 准备持久化目录"
mkdir -p data/runs
touch data/applications.db

echo "==> 启动容器"
docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
docker run -d --name "$CONTAINER" \
  --restart unless-stopped \
  -p "${PORT}:8501" \
  --env-file .env \
  -v "$(pwd)/data/applications.db:/app/data/applications.db" \
  -v "$(pwd)/data/runs:/app/data/runs" \
  "$IMAGE"

echo "==> 等待启动（约 20 秒）"
sleep 20
if curl -s -m 10 "http://localhost:${PORT}/_stcore/health" >/dev/null; then
  echo "    健康检查通过"
else
  echo "    健康检查失败。看日志：docker logs $CONTAINER"
  exit 1
fi

IP=$(curl -s -m 5 ifconfig.me 2>/dev/null || echo "你的服务器IP")
cat <<EOF

部署完成。

  本机访问:  http://localhost:${PORT}
  公网访问:  http://${IP}:${PORT}

用手机打开公网地址，才算真的达成验收。

⚠️ 打不开的话，去云厂商控制台的【安全组】放行 ${PORT} 端口——
   默认只开 22/80/443，这一步忘了 URL 一定打不开。

查看日志:  docker logs -f ${CONTAINER}
停止服务:  docker rm -f ${CONTAINER}
EOF
