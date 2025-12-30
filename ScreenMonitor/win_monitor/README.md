# 数据泄露检测系统 - 监控模块 (Win Monitor)

本模块主要负责监控 Windows 系统下的文件操作行为。

## 安装指南

确保已安装 Python 环境，然后在当前目录下运行以下命令安装依赖：

```bash
pip install -r requirements.txt
```

## 启动指南

### 启动 Web 服务器

在当前目录下 (`win_monitor`) 运行以下命令启动 Web 服务：

```bash
python web_server.py
```

服务启动后，打开浏览器访问：

[http://localhost:5000](http://localhost:5000)
