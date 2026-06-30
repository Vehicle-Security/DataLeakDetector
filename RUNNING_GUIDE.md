# DataLeakDetector 项目完整运行指南

## 1. 环境搭建

### 1.1 创建 Conda 环境

```bash
conda create -n dataleak python=3.10 -y
conda activate dataleak
```

### 1.2 安装依赖

在项目根目录执行：

```bash
pip install -r 01-FrameAnalyzer/file_tracker/requirements.txt
pip install -r 03-LeakReasoner/requirements.txt
pip install opencv-python torch torchvision easyocr thefuzz Pillow moviepy
```

### 1.3 配置大模型 (LLM) 密钥

在项目的 `03-LeakReasoner` 目录下，新建 `.env` 文件，填入你的大模型密钥：

```env
# 必填：大模型 API 密钥
LLM_API_KEY="sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxx"

# 选填：模型名字（默认 qwen-plus）
LLM_MODEL_NAME="qwen-plus" 

# 选填：如果使用第三方兼容接口
# LLM_BASE_URL="https://xxx.xxx.xxx/v1"
```

---

## 2. Datalog 推理引擎说明（已适配 Windows）

### 自动检测机制

项目的核心推理引擎 `DatalogEngine` 现已支持**双引擎自动切换**：

| 环境 | 引擎 | 说明 |
|------|------|------|
| Linux/macOS（已安装 Souffle） | Souffle 原生引擎 | 高性能 C++ 编译执行 |
| Windows / 未安装 Souffle | Python 降级引擎 | 纯 Python 不动点迭代，零依赖 |

**运行时自动检测**，无需手动配置：
- 启动时检测 `souffle` 命令是否可用
- 可用 → 使用 Souffle（显示 `[OK] Datalog 引擎初始化成功 (Souffle)`）
- 不可用 → 自动降级（显示 `[WARN] Souffle 未找到，自动切换为 Python Datalog 引擎`）

两套引擎的**推理结果完全一致**，对外接口完全不变。

### （可选）安装 Souffle 获得最佳性能

如果你在 Linux/macOS 环境下，可以安装 Souffle 以获得更高性能：

```bash
# Ubuntu/Debian
sudo apt-get install souffle

# macOS
brew install souffle
```

---

## 3. 运行方式

### 3.1 验证核心引擎（推荐先执行）

使用项目自带的 Mock 测试数据，验证 LLM + Datalog 推理是否畅通：

```bash
conda activate dataleak
cd 03-LeakReasoner
python test.py
```

**预期输出**：
```
[WARN] Souffle 未找到，自动切换为 Python Datalog 引擎    ← Windows 正常提示
[OK] Datalog 引擎初始化成功 (Python 降级模式)

阶段1: 准备测试数据
   已加载测试场景: 剪贴板泄露

阶段2: LLM 分析生成 Datalog 事实
   ✅ LLM 返回结果
   ✅ 生成了 N 条 Datalog 事实

阶段3: Datalog 推理引擎
   [INFO] 开始 Python Datalog 推理...
   [OK] 发现 N 条泄露路径

阶段4: 检测结果
   🚨 检测到 N 条泄露路径
```

### 3.2 端到端完整流程

准备好你的操作日志 (`.json`) 和录屏视频 (`.mp4`)，执行：

```bash
cd DataLeakDetector-main
python main/run_e2e.py --log "你的日志文件.json" --video "你的录屏文件.mp4"
```

可选参数：
```bash
python main/run_e2e.py --log log.json --video video.mp4 --keywords 机密 合同 工资
```

---

## 4. 常见问题

### Q: Windows 上出现 `Souffle not found` 报错？
**A**: 这是正常现象。引擎会自动降级为 Python 模式，不影响使用。你会看到 `[WARN]` 提示后紧跟 `[OK]`，说明降级成功。

### Q: `LLM_API_KEY` 相关报错？
**A**: 请确保 `03-LeakReasoner/.env` 文件已正确配置你的大模型密钥。

### Q: `main/run_e2e.py` 没有输出结果？
**A**: 请确保：
1. 提供的日志文件 (`.json`) 格式正确，包含 `timestamp`, `file_path`, `event_type` 等字段
2. 日志中存在与默认敏感关键词匹配的文件操作
3. 大模型 API 密钥配置正确
