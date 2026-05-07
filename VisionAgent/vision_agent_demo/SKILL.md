---
name: vision_agent_demo
description: |
  当用户需要基于 GetStream/Vision-Agents 框架创建视频分析应用、特别是视频状态识别类 demo 时，
  使用此 skill。覆盖场景包括：识别视频中物体的状态变化（如烧水、烹饪、运动姿态等）、
  实时视频流分析、YOLO + 多模态 LLM 结合的视频理解、自定义 VideoProcessor 开发。
  即使用户没有明确提到 "Vision-Agents"，只要涉及实时视频 AI 分析、视频状态检测、
  物体行为识别、摄像头智能监控，都应该使用此 skill。
---

# Vision Agent Demo —— 视频状态识别（烧水检测）

## 概述

本 skill 提供一个基于 [GetStream/Vision-Agents](https://github.com/GetStream/Vision-Agents) 框架的完整 demo，演示如何识别视频中**烧水状态的变化**（未加热 -> 加热中 -> 沸腾）。

你可以通过此 skill 快速理解 Vision-Agents 的核心架构，并基于它构建自己的视频状态识别应用。

## 核心概念

Vision-Agents 框架的核心组件：

| 组件 | 作用 |
|------|------|
| **Agent** | 智能体，协调 Edge、LLM、Processor、STT/TTS |
| **Edge** | 视频传输层（WebRTC），负责超低延迟的音视频传输 |
| **Processor** | 视频/音频处理器，在视频流上运行 CV 模型（如 YOLO） |
| **LLM** | 多模态大模型（Gemini/OpenAI），理解视频内容并做出响应 |
| **Events** | 事件系统，用于 Processor 与 Agent 之间的异步通信 |

烧水状态识别的数据流：

```
视频源 → Processor (YOLO检测水壶 + CV特征分析) → 状态机 → Events → Agent → LLM响应
                ↓
         带标注的视频输出
```

## 前置依赖

```bash
# 1. 安装 Vision-Agents 框架
pip install vision-agents

# 2. 安装额外依赖
pip install ultralytics opencv-python numpy

# 3. 如果需要 GPU 加速
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 使用方式

本 skill 提供两种运行模式，根据你的场景选择：

### 模式一：本地视频分析（无需 Stream API）

适合分析本地视频文件或连接本地摄像头，快速验证效果。

**运行步骤：**
1. 读取本 skill 的 `references/boiling_water_processor.py`
2. 运行 `references/demo_local.py`
3. 按 `q` 退出

### 模式二：Vision-Agents Agent 实时流（需要 Stream API）

适合构建生产级的实时视频 AI Agent，支持远程视频通话交互。

**运行步骤：**
1. 到 [Stream](https://getstream.io/try-for-free/) 获取免费 API Key
2. 配置 `.env` 文件：`STREAM_API_KEY=xxx` `STREAM_API_SECRET=xxx`
3. 读取本 skill 的 `references/demo_agent.py`
4. 运行 Agent 并加入视频通话

## 烧水状态识别原理

### 状态定义

| 状态 | 标识 | 视觉特征 |
|------|------|----------|
| 未加热 | `idle` | 水壶静置，无明显运动或蒸汽 |
| 加热中 | `heating` | 底部出现小气泡，轻微蒸汽 |
| 沸腾 | `boiling` | 大量气泡翻滚，明显白色蒸汽上升 |

### 检测算法

1. **目标检测**：使用 YOLO 定位水壶/容器在画面中的位置
2. **ROI 提取**：在水壶上方扩展区域，捕获蒸汽和气泡
3. **蒸汽检测**：帧间差分 + 亮度阈值，检测上升的白色雾气
4. **气泡检测**：拉普拉斯边缘检测，量化水面纹理复杂度
5. **状态判定**：综合蒸汽指数和气泡指数，通过阈值判定当前状态
6. **状态机**：防止状态抖动，要求新状态持续一定时间才确认变化

### 事件系统

Processor 通过事件通知 Agent 状态变化：

- `WaterStateChangedEvent` — 烧水状态发生变化时触发
- `BoilingDetectedEvent` — 检测到沸腾时触发（可用于语音提醒）

## 代码结构

```
vision_agent_demo/
├── SKILL.md                              # 本文件（使用指南）
└── references/
    ├── boiling_water_processor.py        # 核心处理器（CV + 状态机）
    ├── demo_local.py                     # 本地视频/摄像头 demo
    ├── demo_agent.py                     # Vision-Agents Agent 集成 demo
    └── instructions.md                   # Agent 的系统指令模板
```

## 扩展建议

基于本 demo，你可以轻松扩展更多视频状态识别场景：

- **烹饪监控**：识别炒菜、炖煮、煎烤等烹饪阶段
- **运动分析**：识别健身动作的完成度和规范性
- **工业质检**：识别生产线上产品的状态变化
- **安全监控**：识别异常行为（跌倒、打架、入侵）

扩展方法：修改 `BoilingWaterProcessor` 中的：
1. `target_classes` — 修改检测目标类别
2. `analyze_*` 方法 — 自定义视觉特征分析逻辑
3. `determine_state` — 调整状态判定规则
4. 事件类 — 定义新的事件类型

## 完整代码

请参考本 skill 目录下的 `references/` 文件夹获取全部可运行代码。

### 快速预览：核心 Processor 代码

```python
from boiling_water_processor import BoilingWaterProcessor
import asyncio

async def main():
    processor = BoilingWaterProcessor(
        fps=5,
        yolo_model_path="yolo11n.pt",
        steam_threshold=15.0,
        bubble_threshold=30.0,
    )
    
    # 分析本地视频文件
    await processor.process_video("path/to/video.mp4")
    
    # 或使用摄像头
    # await processor.process_video(0)

if __name__ == "__main__":
    asyncio.run(main())
```

### 快速预览：Agent 集成代码

```python
from vision_agents.core import Agent, User
from vision_agents.plugins import gemini, getstream
from boiling_water_processor import BoilingWaterProcessor

async def create_agent(**kwargs):
    processor = BoilingWaterProcessor(fps=5)
    
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="烧水助手"),
        instructions="你是一个智能烧水监控助手...",
        llm=gemini.Realtime(fps=3),
        processors=[processor],
    )
    
    agent.events.merge(processor.events)
    
    @agent.events.subscribe
    async def on_state_changed(event):
        if event.new_state == "boiling":
            await agent.say("水已经烧开了！请关火。")
    
    return agent
```

## 常见问题

**Q: 没有 Stream API Key 能运行吗？**  
A: 可以。使用模式一（本地视频分析）完全不需要 Stream API。只有实时流模式才需要。

**Q: YOLO 模型下载失败怎么办？**  
A: 首次运行时会自动从 Ultralytics 官网下载 `yolo11n.pt`。如果网络受限，可手动下载放到当前目录。

**Q: 检测效果不好怎么优化？**  
A: 调整 `steam_threshold` 和 `bubble_threshold` 参数，或改用多模态 LLM（Gemini）直接分析帧内容，替代纯 CV 方法。

**Q: 可以识别其他状态吗？**  
A: 完全可以。修改 Processor 中的状态定义和分析逻辑即可。参考上文"扩展建议"。
