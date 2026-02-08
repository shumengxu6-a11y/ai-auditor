# TrainPal Short Video Compliance Audit Assistant - PRD (v2.0 Qwen Edition)

## 1. Project Background
为满足出海营销团队每天 1000+ 条短视频的高吞吐审核需求，我们参考了 **AdGuardAi** 的设计理念，重构了 **AI 智能合规审核助手**。新版本核心旨在使用 **国产多模态大模型 (Qwen-VL-Max)** 替代昂贵的海外闭源模型，结合本地轻量化预处理，实现**降本增效**与**政治/文化合规**的双重保障。

## 2. Core Requirements & Tech Stack

### 2.1 核心技术栈 (New Tech Stack)
*   **多模态大脑**: **Aliyun Qwen-VL-Max (通义千问-Max)**。
    *   *选择理由*: 中文语境下对政治敏感内容（如“一个中国”原则）理解更精准；图文理解能力接近 GPT-4o 但成本更低。
*   **听觉审核**: **OpenAI Whisper (Base Model)** [Local Deployment]。
    *   *选择理由*: 本地运行无需 API 费用，对中英混合语音识别准确率极高。
*   **关键帧引擎**: **OpenCV-Python-Headless**。
    *   *策略*: 放弃高耗能的逐帧分析，采用 **每 5 秒提取关键帧** (Keyframe Extraction Strategy)，大幅降低 Token 消耗。

### 2.2 核心业务红线 (Asset A)
系统内置硬性合规逻辑 (`trainpal_rules.json`)：
1.  **竞品排他 [CRITICAL]**: 严禁出现 LNER, Avanti 等竞品 Logo。
2.  **政治敏度 [CRITICAL]**: 严禁将“香港”与“中国”在语音/字幕中并列表述，必须符合一个中国原则。
3.  **价格合规 [HIGH]**: 严禁绝对化用词 ("Cheapest", "No.1")，除非有明确 T&Cs。
4.  **画面风险 [HIGH]**: 严禁不文明行为 (踢箱子、脚踩座椅)。

## 3. System Architecture & Workflow

### 3.1 处理流程 (Pipeline)
1.  **Ingest**: 用户上传 MP4 视频 (UI: Streamlit)。
2.  **Pre-processing (Local)**:
    *   **Visual**: OpenCV 每隔 5 秒抽样一张高清关键帧 (30s 视频 -> 6 张图)。
    *   **Audio**: Whisper(Local) 提取全量语音文本 (Transcript)。
3.  **Inference (Cloud)**:
    *   将 `[图片列表 + 语音文本 + System Prompt]` 打包发送至 **Qwen-VL-Max**。
    *   模型进行**跨模态交叉验证** (例如：画面有 Logo 但语音未提及，或语音承诺退款但画面显示不可退)。
4.  **Report**: 解析 JSON 结果，渲染 **AdGuard 风格** 仪表盘。

### 3.2 成本与性能模型 (Cost Model)
*   **策略**: 关键帧采样 (Keyframe Sampling)。
*   **计算公式**:
    *   Visual: 6 Frames * ~1000 Tokens/Frame = 6,000 Tokens
    *   Audio: ~500 Tokens (Text)
    *   Total: ~6.5k Tokens / Video.
*   **预估单价**: Qwen-VL-Max (按 Input Token 计费，假设 ¥0.02/1k Tokens?) -> **约 ¥0.13 / 视频**。
*   *对比*: 相比 GPT-4o 视频流处理，成本降低约 **85%**。

## 4. UI/UX Specification (AdGuard Style)

1.  **Side Panel (Control)**:
    *   API Key 安全输入。
    *   规则库预览 (DataFrame)。
    *   参数微调 (采样间隔)。
2.  **Main Dashboard**:
    *   **Status Indicator**: 动态步骤条 (Extracting -> Transcribing -> Auditing)。
    *   **Result Card**: 大字体 PASS (绿色) / FAIL (红色) 状态卡片。
    *   **Details**: 违规详情折叠面板 (Expander)，精确到时间戳。
    *   **Metrics**: 实时显示本次消耗 Token 及费用。

## 5. Deployment Guide
1.  Install: `pip install -r requirements.txt` (Includes `dashscope`, `openai-whisper`, `opencv-python-headless`).
2.  Run: `streamlit run app.py`.

---
**Version**: 2.0.0 (Qwen-VL Edition)
**Date**: 2026-02-06
