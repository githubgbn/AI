"""
Vision-Agents Agent 集成 Demo —— 实时烧水监控助手

运行方式：
    1. 配置环境变量（或创建 .env 文件）：
        STREAM_API_KEY=your_api_key
        STREAM_API_SECRET=your_api_secret
    2. 运行 Agent：
        python demo_agent.py
    3. 根据控制台提示，使用 Stream 客户端加入视频通话

特性：
- 实时分析远程视频流中的烧水状态
- 水烧开时自动语音提醒
- 支持用户语音询问当前状态
- 可扩展为智能家居场景（自动关火、记录烧水日志等）
"""

import logging
from typing import Any, Dict

from dotenv import load_dotenv

from vision_agents.core import Agent, Runner, User
from vision_agents.core.agents import AgentLauncher
from vision_agents.plugins import deepgram, elevenlabs, gemini, getstream

from boiling_water_processor import (
    BoilingDetectedEvent,
    BoilingWaterProcessor,
    WaterStateChangedEvent,
)

load_dotenv()
logger = logging.getLogger(__name__)


async def create_agent(**kwargs) -> Agent:
    """创建烧水监控 Agent。"""

    # 1. 创建烧水状态识别处理器
    boiling_processor = BoilingWaterProcessor(
        fps=5,
        yolo_model_path="yolo11n.pt",
        device="cpu",
        steam_threshold=15.0,
        bubble_threshold=30.0,
        state_change_min_duration=3.0,
    )

    # 2. 配置多模态 LLM（Gemini Realtime）
    llm = gemini.Realtime(fps=3)

    # 3. 配置语音交互（可选）
    stt = deepgram.STT(eager_turn_detection=True)
    tts = elevenlabs.TTS()

    # 4. 创建 Agent
    agent = Agent(
        edge=getstream.Edge(),
        agent_user=User(name="烧水助手", id="boiling-agent"),
        instructions="Read @instructions.md",
        llm=llm,
        processors=[boiling_processor],
        stt=stt,
        tts=tts,
    )

    # 5. 合并处理器事件到 Agent 事件系统
    agent.events.merge(boiling_processor.events)

    # 6. 注册 LLM 可调用的工具函数
    @llm.register_function(
        description="获取当前烧水状态及历史变化记录。"
    )
    async def get_boiling_status() -> Dict[str, Any]:
        history = boiling_processor.state_history
        current = boiling_processor.current_state
        duration = __import__("time").time() - boiling_processor.state_start_time
        return {
            "current_state": current,
            "state_duration_seconds": round(duration, 1),
            "total_changes": len(history),
            "history": [
                {
                    "state": h["state"],
                    "timestamp": __import__("time").strftime(
                        "%H:%M:%S", __import__("time").localtime(h["timestamp"])
                    ),
                    "duration": round(h["duration"], 1),
                    "confidence": round(h["confidence"], 2),
                }
                for h in history
            ],
        }

    @llm.register_function(
        description="获取最后一次状态变化的详细信息。"
    )
    async def get_last_state_change() -> Dict[str, Any]:
        history = boiling_processor.state_history
        if not history:
            return {"message": "暂无状态变化记录"}
        last = history[-1]
        return {
            "from_state": last.get("old_state", "unknown"),
            "to_state": last["state"],
            "timestamp": __import__("time").strftime(
                "%H:%M:%S", __import__("time").localtime(last["timestamp"])
            ),
            "duration": round(last["duration"], 1),
            "confidence": round(last["confidence"], 2),
        }

    # 7. 订阅处理器事件，实现自动语音提醒
    @agent.events.subscribe
    async def on_water_state_changed(event: WaterStateChangedEvent):
        """状态变化时记录日志。"""
        logger.info(
            f"事件: 烧水状态变化 {event.old_state} -> {event.new_state} "
            f"(置信度: {event.confidence:.2f})"
        )

    @agent.events.subscribe
    async def on_boiling_detected(event: BoilingDetectedEvent):
        """检测到沸腾时，语音提醒用户。"""
        msg = f"水已经烧开了！从加热到沸腾用了 {event.duration_seconds:.0f} 秒。"
        logger.info(f"🔔 {msg}")
        await agent.say(msg)

    return agent


async def join_call(agent: Agent, call_type: str, call_id: str, **kwargs) -> None:
    """Agent 加入视频通话。"""
    call = await agent.create_call(call_type, call_id)

    async with agent.join(call):
        # 初始化问候语
        await agent.llm.simple_response(
            text="你好，我是你的智能烧水助手。我会实时监控烧水状态，水烧开时会提醒你。"
        )

        # 持续运行直到通话结束
        await agent.finish()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    Runner(AgentLauncher(create_agent=create_agent, join_call=join_call)).cli()
