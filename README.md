# DataLeakDetector
## 项目介绍
一个基于视频分析和LLM推理的数据泄漏检测系统，能够自动识别视频中的敏感信息并进行智能判断。
## quickstart
### 调用api
填写要使用的模型名称、apikey、url
### 本地运行
开启vllm服务或ollama服务
### 启动app
python app.py

# DataLeakDetector

## 项目介绍

一个基于视频分析和LLM推理的数据泄漏检测系统，能够自动识别视频中的敏感信息并进行智能判断。

## quickstart

### 调用api

填写要使用的模型名称、apikey、url

### 本地运行

开启vllm服务或ollama服务

### 启动app

python [app.py](http://app.py)

## 🚀 11.29 更新 - KeyFrame文件夹

### ✨ 新增功能

- **KeyFrame 文件夹**: 用于从视频中自动识别各个敏感操作的起止时间

### 🎮 快速开始

设置好视频文件路径后直接运行：

> python [main.py](http://main.py)

### 输出示例

> 🎬 开始处理整个视频...
> 🔍 寻找操作边界...
> 🤖 开始VLM敏感操作原子级分析...
> 🔄 开始操作边界扩展...
>
> 📁 处理第 1 个场景:
> 🏷️ 应用: 飞书
> ⚡ 操作: 即时通讯发送\
> ⏰ 时间区间: 76.7s - 94.4s
> 📸 找到 6 个关键帧
> ✅ 成功保存到: scene_01_飞书_即时通讯发送
>
> 📁 处理第 2 个场景:
> 🏷️ 应用: Kimi
> ⚡ 操作: AI交互
> ⏰ 时间区间: 169.1s - 179.0s
> 📸 找到 3 个关键帧
> ✅ 成功保存到: scene_02_Kimi_AI交互
>
> 📊 场景关键帧保存完成:
> 📈 总场景数: 2
> ✅ 成功保存: 2
> 📁 保存位置: ./output/scene_keyframes


