# PPTtoEdit Web 部署包

这个文件夹是一个独立的云端网页版部署包，适合上传到腾讯云 CVM 后用 Docker Compose 启动。

## 功能

- 上传 `.pdf`、`.pptx`、`.ppt`
- PDF 自动转成图片型 PPT
- 云端 OCR 识别每页文字框
- 网页中查看页面预览、移动/缩放识别框、修改文字
- 导出可编辑 `.pptx`
- 后台队列处理耗时任务，浏览器只负责上传、预览和下载

## 推荐服务器

测试阶段建议腾讯云 CVM：

- Ubuntu Server 22.04 LTS
- 16 核 32G 内存更稳，最低可先试 8 核 16G
- 系统盘 100G，数据盘 200G 起
- 安全组开放 22、80；正式 HTTPS 再开放 443

这个项目依赖 OCR、Paddle、IOPaint，Docker 镜像会比较大，第一次构建可能需要较长时间。

## 上传到腾讯云

在服务器上准备目录：

```bash
sudo mkdir -p /opt/ppttoedit-web
sudo chown -R $USER:$USER /opt/ppttoedit-web
```

把整个 `web_deploy` 文件夹内容上传到 `/opt/ppttoedit-web`。

如果用 `scp`，可以在本机 PowerShell 中执行：

```powershell
scp -r E:\projects\PPTtoEdit\web_deploy\* ubuntu@你的服务器IP:/opt/ppttoedit-web/
```

## 安装 Docker

登录服务器：

```bash
ssh ubuntu@你的服务器IP
```

安装 Docker：

```bash
sudo apt update
sudo apt install -y curl
curl -fsSL https://get.docker.com | sudo bash
sudo usermod -aG docker $USER
newgrp docker
docker --version
docker compose version
```

## 启动

```bash
cd /opt/ppttoedit-web
docker compose up -d --build
docker compose ps
```

访问：

```text
http://你的服务器IP
```

查看日志：

```bash
docker compose logs -f api
docker compose logs -f worker
```

停止：

```bash
docker compose down
```

## Windows 本地开发运行

如果只是本机预览和测试 Web 版，不一定需要 Docker。项目内置了一个开发脚本，会启动 FastAPI 后端和静态前端，并复用桌面版已经安装好的 `.py310deps`、`.py310iopaint` 依赖目录。

在 PowerShell 中运行：

```powershell
cd E:\projects\PPTtoEdit
.\web_deploy\run_local_dev.ps1
```

脚本会打开：

```text
http://127.0.0.1:5173
```

后端地址是：

```text
http://127.0.0.1:8000
```

保持 PowerShell 窗口打开即可。停止时按 `Ctrl+C`。

## 文件保存位置

用户上传文件、工作目录、导出结果会保存在：

```text
/opt/ppttoedit-web/data/jobs
```

正式上线前建议加一个定时清理任务，删除超过 24 小时或 7 天的任务文件，避免磁盘被占满。

## HTTPS 和域名

MVP 可以先用公网 IP 访问。正式上线建议：

1. 域名完成备案。
2. 申请腾讯云 SSL 证书。
3. 在 Nginx 或腾讯云负载均衡上配置 HTTPS。
4. 安全组只保留 22、80、443。

## 注意

- 第一次导出可编辑 PPT 时，IOPaint 可能会下载 `lama` 模型，耗时会更久。
- 如果服务器没有足够内存，OCR 或导出可能失败，建议先用 16G 以上内存测试。
- 当前 Web 版是单机部署包，适合 MVP。用户量上来后，应把文件存储迁移到腾讯云 COS，把 Worker 横向扩容。
