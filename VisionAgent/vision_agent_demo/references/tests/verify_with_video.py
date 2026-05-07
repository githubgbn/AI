"""
使用模拟视频验证烧水状态识别效果（无头模式，无需显示器）

运行方式：
    cd vision_agent_demo/references
    python3 tests/verify_with_video.py

流程：
    1. 调用 generate_test_video.py 生成测试视频（如不存在）
    2. 使用 BoilingWaterProcessor 逐帧分析
    3. 打印每帧状态到控制台
    4. 保存带标注的输出视频到 tests/output_annotated.mp4
    5. 输出完整状态历史报告
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from boiling_water_processor import BoilingWaterProcessor


VIDEO_PATH = os.path.join(os.path.dirname(__file__), "test_boiling.mp4")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output_annotated.mp4")


def ensure_video_exists():
    """确保测试视频已生成。"""
    if not os.path.exists(VIDEO_PATH):
        print("测试视频不存在，正在生成...")
        import generate_test_video
        generate_test_video.main()


def main():
    ensure_video_exists()

    print("=" * 60)
    print("视频验证：烧水状态识别")
    print("=" * 60)
    print(f"输入视频: {VIDEO_PATH}")
    print(f"输出视频: {OUTPUT_PATH}")
    print("=" * 60)

    # 创建处理器
    processor = BoilingWaterProcessor(
        fps=5,
        yolo_model_path="yolo11n.pt",
        device="cpu",
        steam_threshold=10.0,
        bubble_threshold=20.0,
        state_change_min_duration=0.5,
    )

    # 加载模型
    processor._ensure_model()

    # 初始化状态起始时间为 0，使后续可用视频时间戳进行防抖
    processor.state_start_time = 0.0

    # 打开输入视频
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 {VIDEO_PATH}")
        return

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS) or 5
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 创建输出视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))
    if not writer.isOpened():
        print("警告: 无法创建 MP4 输出，尝试 AVI...")
        OUTPUT_PATH_AVI = OUTPUT_PATH.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(OUTPUT_PATH_AVI, fourcc, fps, (width, height))

    print(f"\n视频信息: {width}x{height} @ {fps:.1f}fps, 共 {total_frames} 帧")
    print("\n开始处理...")
    print("-" * 60)
    print(f"{'帧':>4} | {'时间':>6} | {'蒸汽':>6} | {'气泡':>6} | {'状态':>10}")
    print("-" * 60)

    prev_roi = None
    frame_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 检测容器（YOLO）
            bbox = processor.detect_container(frame)
            yolo_detected = bbox is not None

            # 提取 ROI：优先使用 YOLO 检测框，否则使用固定区域覆盖壶身及上方
            roi = None
            if bbox:
                x1, y1, x2, y2 = bbox
                margin = int((y2 - y1) * 0.6)
                roi_y1 = max(0, y1 - margin)
                roi = frame[roi_y1:y2, x1:x2]
            else:
                # YOLO 未检测到容器时，使用固定 ROI 覆盖水壶及上方蒸汽区域
                # 测试视频中水壶位置大致在 (240,280)-(400,420)，上方扩展到 y=150
                roi = frame[150:420, 220:420]
                bbox = (220, 150, 420, 420)  # 用于标注框显示

            # 分析特征
            steam_score = 0.0
            bubble_score = 0.0
            if roi is not None and roi.size > 0:
                steam_score = processor.analyze_steam(roi, prev_roi)
                bubble_score = processor.analyze_bubbles(roi)
                prev_roi = roi.copy()

            # 判定状态（使用视频时间戳进行防抖，避免处理速度过快导致状态被吞）
            new_state = processor.determine_state(steam_score, bubble_score)
            confidence = min(1.0, (steam_score + bubble_score) / 100.0)
            video_timestamp = frame_idx / fps
            processor.update_state(new_state, confidence, timestamp=video_timestamp)

            # 标注
            annotated = processor.annotate_frame(
                frame, bbox, processor.current_state, steam_score, bubble_score
            )

            # 写入输出视频
            writer.write(annotated)

            # 打印进度
            time_sec = frame_idx / fps
            print(
                f"{frame_idx:4d} | {time_sec:5.1f}s | {steam_score:6.2f} | "
                f"{bubble_score:6.2f} | {processor.current_state:>10}"
            )

            frame_idx += 1

    finally:
        cap.release()
        writer.release()

    print("-" * 60)
    print(f"\n处理完成！共处理 {frame_idx} 帧")
    print(f"标注视频已保存: {OUTPUT_PATH}")

    # 输出状态历史
    print("\n" + "=" * 60)
    print("状态变化历史")
    print("=" * 60)
    history = processor.state_history
    if not history:
        print("  无状态变化")
    else:
        for i, h in enumerate(history, 1):
            print(
                f"  [{i}] 视频 {h['timestamp']:.1f}s | {h['state']:8s} | "
                f"持续 {h['duration']:.1f}s | 置信度 {h['confidence']:.2f}"
            )

    # 最终验证
    print("\n" + "=" * 60)
    print("验证结果")
    print("=" * 60)

    expected_transitions = ["heating", "boiling"]
    actual_transitions = [h["state"] for h in history]

    if actual_transitions == expected_transitions:
        print("✅ 状态变化序列正确: idle -> heating -> boiling")
    else:
        print(f"⚠️  状态变化序列: idle -> {' -> '.join(actual_transitions) if actual_transitions else '(无变化)'}")
        print(f"   期望: idle -> heating -> boiling")

    if history and history[-1]["state"] == "boiling":
        print("✅ 最终状态为 boiling")
    else:
        print("⚠️  最终状态不为 boiling")

    print("=" * 60)


if __name__ == "__main__":
    main()
