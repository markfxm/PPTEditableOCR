# 腾讯云部署步骤

1. 购买腾讯云 CVM，系统选择 Ubuntu Server 22.04 LTS。
2. 安全组开放 `22` 和 `80`，正式 HTTPS 再开放 `443`。
3. SSH 登录服务器。
4. 安装 Docker。
5. 上传本文件夹全部内容到 `/opt/ppttoedit-web`。
6. 执行 `docker compose up -d --build`。
7. 浏览器打开 `http://服务器公网IP`。

常用命令：

```bash
cd /opt/ppttoedit-web
docker compose ps
docker compose logs -f worker
docker compose restart
docker compose down
```

如果构建或运行太慢，优先升级 CPU/内存。确认业务跑通后，再考虑 GPU Worker。
