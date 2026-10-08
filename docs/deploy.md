# 部署到云服务器

目标：拿到一个**能在手机上打开**的公网 URL。

用 `http://服务器IP:8501` 访问**不需要备案**——备案只针对绑定域名。

## 0. 服务器配置建议

| 项 | 建议 | 原因 |
|---|---|---|
| 厂商 | 阿里云 / 腾讯云**学生机** | 学生认证后约 ¥99/年 |
| CPU/内存 | 2 核 2G 起 | 运行时只吃 **218MB**（已实测，含模型） |
| **磁盘** | **≥ 40GB** | 镜像 3.15GB + 构建缓存 ~2.6GB + 系统 |
| 系统 | Ubuntu 22.04 / 24.04 | Docker 生态最省事 |

## 1. 服务器装 Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER    # 之后重新登录，免得每条命令都要 sudo
docker --version
```

## 2. 把代码传上去

```bash
git clone https://github.com/X-Shadow/job-agent.git
cd job-agent
```

## 3. ⚠️ 单独传模型（92MB，不在 git 里）

模型在 `.gitignore` 里，**服务器上下载不到**——HuggingFace 在国内不通
（本地那次 178 秒的重试就是证据）。所以必须单独传。

在你自己的电脑上：

```bash
# 如果还没暂存过
python scripts/stage_model.py

# 传上去（把 user/服务器IP 换成你的）
scp -r models user@服务器IP:~/job-agent/
```

## 4. 配置密钥

```bash
cp .env.example .env   # 没有 .env.example 就直接新建
nano .env
```

```ini
DEEPSEEK_API_KEY=你的key
# 走本地 embedding 时下面这行可以留空
ZHIPU_API_KEY=
```

**密钥只放在服务器上，绝不进 git。**

## 5. 构建镜像

国内服务器直连官方源常常很慢或不通，**用国内镜像**：

```bash
docker build -t job-agent \
  --build-arg PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple \
  --build-arg TORCH_INDEX=https://mirror.sjtu.edu.cn/pytorch-wheels/cpu \
  .
```

> 这两个源是国内实测可用的。
> ⚠️ **别用 `mirrors.aliyun.com/pytorch-wheels/cpu`**——它能打开，但**不是合法的 pip 索引**，
> `pip index versions` 会报 `No matching distribution found`。

## 6. 启动

```bash
mkdir -p data/runs && touch data/applications.db   # 供挂载用，先建好

docker run -d --name job-agent \
  --restart unless-stopped \
  -p 8501:8501 \
  --env-file .env \
  -v $(pwd)/data/applications.db:/app/data/applications.db \
  -v $(pwd)/data/runs:/app/data/runs \
  job-agent
```

**为什么要挂载那两个**：镜像里的 `/app/data/` 是临时的，容器一重建，
你的投递记录和分析历史就没了。挂到宿主机上才留得住。

`--restart unless-stopped`：服务器重启后容器自动起来。

## 7. ⚠️ 开放端口

去云厂商控制台的**安全组**放行 **8501**。默认只开 22/80/443，
不开这个，外网连不上——**这一步忘了，URL 一定打不开**。

## 8. 验证

```bash
curl http://localhost:8501/_stcore/health     # 服务器上自测，应返回 ok
```

然后**用手机**打开 `http://服务器IP:8501`（用手机验证，才算真的达成验收）。

## 常见问题

| 现象 | 原因 | 处理 |
|---|---|---|
| 手机上打不开，服务器上 curl 正常 | 安全组没放行 8501 | 回第 7 步 |
| 服务器上 curl 也不通 | 容器没起来 | `docker logs job-agent` |
| 页面能开，但一点分析就报错 | `.env` 没配或密钥错 | 看日志里是不是缺 `DEEPSEEK_API_KEY` |
| 构建卡在下载 torch | 服务器连不上官方源 | 确认加了 `--build-arg TORCH_INDEX=...` |
| 构建报 `No matching distribution found for torch` | 用了非法的索引 URL | 换成交大那个（见第 5 步） |
| 内存不足 / 进程被杀 | 服务器内存太小 | 2G 起；再不行切 `EMBED_BACKEND=zhipu`（镜像瘦 1.5GB） |

## 更新代码后

```bash
git pull
docker build -t job-agent --build-arg ... .   # 同样的构建命令
docker rm -f job-agent && docker run -d ...   # 同样的启动命令
```

改了 `data/experience.json` 也要重新 build——**向量索引是构建期生成的**。
