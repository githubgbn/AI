"""
本地视频/摄像头烧水状态识别 Demo

运行方式：
    python demo_local.py                  # 使用默认摄像头 (索引 0)
    python demo_local.py video.mp4        # 分析本地视频文件
    python demo_local.py http://xxx.m3u8  # 分析网络视频流

特性：
- 无需 Stream API Key，完全离线运行
- 支持视频文件、摄像头、网络流
- 实时显示状态面板和检测框
- 退出时打印完整状态变化历史
"""

import asyncio
import logging
import sys

from boiling_water_processor import BoilingWaterProcessor


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    # 解析视频源参数
    if len(sys.argv) > 1:
        source = sys.argv[1]
        # 尝试转为整数（摄像头索引）
        try:
            source = int(source)
        except ValueError:
            pass  # 保持字符串（文件路径）
    else:
        source = 0  # 默认使用第一个摄像头

    print("=" * 50)
    print("烧水状态识别 Demo")
    print("=" * 50)
    print(f"视频源: {source}")
    print("按 'q' 键退出")
    print("=" * 50)

    # 创建处理器（可根据实际场景调整参数）
    processor = BoilingWaterProcessor(
        fps=5,                          # 每秒处理 5 帧（平衡性能与效果）
        yolo_model_path="yolo11n.pt",   # YOLO 模型（首次运行自动下载）
        device="cpu",                   # 运行设备（cpu / cuda）
        steam_threshold=15.0,           # 蒸汽判定阈值（越大越严格）
        bubble_threshold=30.0,          # 气泡判定阈值（越大越严格）
        state_change_min_duration=3.0,  # 状态最少持续 3 秒（防抖）
    )

    # 启动异步处理
    asyncio.run(processor.process_local_video(source))


if __name__ == "__main__":
    main()
