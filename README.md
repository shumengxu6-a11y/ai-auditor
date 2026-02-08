# 🛡️ AI 智能视频内容审核助手 (Qwen/DeepSeek Edition)

> **2026 Multimodal Content Moderation System**
> 
> 基于阿里云 Qwen-VL 与 DeepSeek-V3 的多模态短视频审核系统，专为高并发、高精度的内容安全场景设计。

## 📖 项目简介 (Introduction)

本项目是一个 **AI Native** 的非结构化数据审核平台。针对传统审核系统“看不懂画面、听不懂语境”的痛点，利用最新的 **Vision-Language Models (VLM)** 技术，实现了对短视频画面、音频、字幕的**全模态理解**。

系统不仅能识别传统的色情/暴力违规，更能理解**“竞品拉踩”、“虚假宣传”、“政治隐喻”**等高阶语义风险，并提供**秒级**的审核报告与**精确到分**的成本核算。

## ✨ 核心亮点 (Key Features)

*   **🧠 多模态深度理解**: 支持 **DeepSeek-VL2**, **Qwen-VL-Max**, **Gemini 3.0** 等顶尖模型，精准识别视频中的视觉与听觉风险。
*   **💰 实时成本看板**: 内置 **FinOps** 模块，基于 DataLearner 2026 最新定价，实时计算 Token 消耗与人民币成本（低至 ¥0.01/条）。
*   **🛡️ 智能红线体系**: 内置 12 条行业标准业务红线（竞品、广告法、未成年保护等），支持 Chain-of-Thought (CoT) 推理。
*   **⚡ 高可用架构**: 实现模型级 **Failover (自动熔断/降级)** 机制，当主模型 (DeepSeek) 不可用时自动切换至备选模型 (Qwen) 保障服务连续性。
*   **📊 完整交互体验**: 包含视频抽帧预览、违规时间轴定位、Json 原文透视及管理后台看板。

## 🛠️ 技术栈 (Tech Stack)

*   **Frontend**: Streamlit (Python) - 快速构建响应式 Web UI
*   **Core AI**: 
    *   Vision: Qwen-VL-Max / DeepSeek-VL2-Pro
    *   Costing: Real-time Pricing Engine (DataLearner API 2026)
*   **Processing**: OpenCV (智能抽帧/压缩), Base64 Pipeline
*   **Governance**: Local JSON Storage (审计日志与成本追踪)

## 🚀 快速开始 (Quick Start)

### 1. 环境准备
确保您的本地环境已安装 Python 3.8+ 及 **FFmpeg** (用于音频处理)。

> **⚠️ 注意**: 原生 Whisper 模型依赖 FFmpeg 进行音频解码。请务必安装，否则音频审核功能将降级。

*   **Mac**: `brew install ffmpeg`
*   **Windows**: `winget install Gyan.FFmpeg`
*   **Linux**: `sudo apt install ffmpeg`

```bash
# 解压并进入项目目录
cd AI_Video_Auditor_Demo_XuShumeng_v1.0

# 创建虚拟环境 (推荐)
python -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行系统
```bash
streamlit run app.py
```

### 3. 配置 API Key
启动后，在网页左侧侧边栏输入您的 **DashScope API Key** (用于 Qwen) 或 **OpenAI-Compatible Key** (用于 DeepSeek/Gemini)。

> **注意**: 为保障安全，API Key 不会保存在代码中，仅在运行时存储于内存。

## 📂 目录结构

```text
├── app.py                # 主程序入口 (UI & 业务逻辑)
├── auditor_qwen.py       # 核心审核引擎 (API 调用与图像处理)
├── utils.py              # 工具函数库
├── requirements.txt      # 依赖列表
├── cost_history.json     # [自动生成] 成本审计日志
└── README.md             # 项目文档
```

## ⚖️ 免责声明
本项目仅供演示与学习使用，审核结果仅供参考，生产环境建议配合人工复核流程。
