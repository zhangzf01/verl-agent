# VWA Sandbox 使用指南

VWA（VisualWebArena）Reddit 环境运行在 Apptainer sandbox 中，包含 nginx + php-fpm + postgres，对外暴露端口 9999。

## 快速上手

所有操作通过 `scripts/vwa_service.sh` 管理：

```bash
bash scripts/vwa_service.sh start     # 启动
bash scripts/vwa_service.sh stop      # 优雅停止
bash scripts/vwa_service.sh restart   # 重启
bash scripts/vwa_service.sh status    # 查看状态
bash scripts/vwa_service.sh reset-wal # 修复 postgres WAL 损坏（见下）
```

服务起来后访问 `http://localhost:9999`，账号 `MarvelsGrantMan136 / test1234`。

## 注意事项

### 两台机器 localhost 不互通

- `dt-login03`（login 节点）和 `gpue02`（GPU 节点）是不同的机器
- sandbox 必须在**训练所在的机器**上启动
- `/projects/bghp/jguo14/` 是共享存储，两台机器都能读写 sandbox 文件
- 如果在 gpue02 上训练，就在 gpue02 上启动 sandbox

### 正确停止服务（避免 WAL 损坏）

**一定要用 `vwa_service.sh stop`**，不要直接 `pkill supervisord`。

```bash
# 正确
bash scripts/vwa_service.sh stop

# 错误（postgres 没有机会写 checkpoint，下次启动会有 WAL 损坏）
pkill -TERM supervisord
```

### postgres 启动慢

首次启动后 postgres 需要 10-30 秒恢复，期间 HTTP 返回 500。`vwa_service.sh start` 会自动等待最多 120 秒。

## 常见问题

### HTTP 500

postgres 还在启动中，等一下：
```bash
bash scripts/vwa_service.sh status
```

### ERR_CONNECTION_REFUSED (Playwright)

sandbox 没在当前机器上运行。确认 `pgrep -a supervisord` 有输出，且 `curl http://localhost:9999` 返回 200。

### postgres WAL 损坏

症状：日志出现 `could not locate a valid checkpoint record` 或 `invalid resource manager ID`，postgres 反复重启。

原因：上次 supervisord 被直接 kill，postgres 没有写 checkpoint。

修复：
```bash
bash scripts/vwa_service.sh reset-wal
```

这会自动：删除 `postmaster.pid` → 运行 `pg_resetwal -f` → 重启服务。

### 重建 sandbox（数据完全丢失时）

如果 postgres 数据无法恢复，需要从原始 Docker 镜像重建：

```bash
# 1. 下载镜像（51GB，约 30 分钟）
curl -L --retry 5 --retry-delay 10 -C - \
    http://metis.lti.cs.cmu.edu/webarena-images/postmill-populated-exposed-withimg.tar \
    -o /projects/bghp/jguo14/postmill.tar

# 2. 构建 sandbox
APPTAINER_CACHEDIR=/work/nvme/bghp/jguo14/.apptainer_cache \
    apptainer build --sandbox /projects/bghp/jguo14/vwa-reddit-sandbox \
    docker-archive:///projects/bghp/jguo14/postmill.tar

# 3. 删除下载的 tar（节省空间）
rm /projects/bghp/jguo14/postmill.tar
```

## 内部结构

```
/projects/bghp/jguo14/vwa-reddit-sandbox/   # Apptainer writable sandbox
  usr/local/pgsql/data/                       # postgres 数据目录
  var/www/html/                               # Postmill PHP 应用
  etc/supervisor.d/                           # nginx / php-fpm / postgres 进程配置
  etc/nginx/conf.d/default.conf               # nginx 监听 9999，fastcgi 到 19000
```

postgres 连接信息：`pgsql://postmill:postmill@localhost:5432/postmill`
