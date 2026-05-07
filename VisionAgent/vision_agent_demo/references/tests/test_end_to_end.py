"""
端到端模拟测试：模拟完整视频序列，验证状态识别流程

运行方式：
    cd vision_agent_demo/references
    python3 tests/test_end_to_end.py
"""

import os
import sys
import time

# 将父目录加入路径，确保能导入 boiling_water_processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from boiling_water_processor import BoilingWaterProcessor


def test_idle_to_boiling_sequence():
    """
    模拟从 idle -> heating -> boiling 的完整烧水过程。
    共 12 帧，每帧间隔 200ms（模拟 5fps）：
      - 帧 0-3: 平静水面（idle）
      - 帧 4-7: 轻微蒸汽（heating）
      - 帧 8-11: 大量蒸汽+气泡（boiling）
    """
    print("=" * 50)
    print("端到端测试: idle -> heating -> boiling")
    print("=" * 50)

    p = BoilingWaterProcessor(
        steam_threshold=10.0,
        bubble_threshold=20.0,
        state_change_min_duration=0.5,
    )

    np.random.seed(42)
    frames = []

    # 生成模拟帧
    for i in range(12):
        frame = np.full((480, 640, 3), 80, dtype=np.uint8)
        cv2.rectangle(frame, (200, 250), (350, 400), (100, 80, 60), -1)
        roi = frame[150:400, 200:350].copy()

        if i < 4:
            # idle: 平静水面
            pass
        elif i < 8:
            # heating: 少量蒸汽 + 轻微纹理
            roi[50:100, 50:100] = [200, 200, 200]
            roi = (
                roi.astype(np.int16)
                + np.random.randint(-8, 9, roi.shape)
            ).clip(0, 255).astype(np.uint8)
        else:
            # boiling: 大量蒸汽 + 强纹理
            roi[20:150, 30:120] = [240, 240, 240]
            roi = (
                roi.astype(np.int16)
                + np.random.randint(-25, 26, roi.shape)
            ).clip(0, 255).astype(np.uint8)

        frames.append(roi)

    # 逐帧处理
    prev_roi = None
    for i, roi in enumerate(frames):
        steam = p.analyze_steam(roi, prev_roi)
        bubble = p.analyze_bubbles(roi)
        state = p.determine_state(steam, bubble)
        p.update_state(state, min(1.0, (steam + bubble) / 100))

        print(
            f"帧{i:2d}: steam={steam:6.2f}, bubble={bubble:6.2f}, "
            f"state={p.current_state:8s}"
        )
        prev_roi = roi.copy()
        time.sleep(0.2)  # 模拟 5fps 帧间隔

    # 验证结果
    print(f"\n最终状态: {p.current_state}")
    print(f"状态历史记录数: {len(p.state_history)}")
    for h in p.state_history:
        print(
            f"  {h['state']:8s} - 持续 {h['duration']:.1f}s - "
            f"置信度 {h['confidence']:.2f}"
        )

    # 断言
    assert len(p.state_history) >= 2, (
        f"应至少有两次状态变化, 实际 {len(p.state_history)}"
    )
    assert p.state_history[-1]["state"] == "boiling", "最终应为 boiling 状态"

    print("\n✅ 端到端测试通过!")
    return True


def test_constant_boiling():
    """
    模拟持续沸腾状态，验证稳定蒸汽不会被漏检。
    """
    print("\n" + "=" * 50)
    print("端到端测试: 持续沸腾（稳定蒸汽检测）")
    print("=" * 50)

    p = BoilingWaterProcessor(
        steam_threshold=10.0,
        bubble_threshold=20.0,
        state_change_min_duration=0.3,
    )

    np.random.seed(99)
    prev_roi = None

    for i in range(8):
        # 每帧都有大量亮白色区域（模拟稳定蒸汽）
        roi = np.full((100, 100, 3), 100, dtype=np.uint8)
        roi[10:90, 10:90] = [230, 230, 230]
        roi = (
            roi.astype(np.int16)
            + np.random.randint(-10, 11, roi.shape)
        ).clip(0, 255).astype(np.uint8)

        steam = p.analyze_steam(roi, prev_roi)
        bubble = p.analyze_bubbles(roi)
        state = p.determine_state(steam, bubble)
        p.update_state(state, min(1.0, (steam + bubble) / 100))

        print(
            f"帧{i:2d}: steam={steam:6.2f}, bubble={bubble:6.2f}, "
            f"state={p.current_state:8s}"
        )
        prev_roi = roi.copy()
        time.sleep(0.15)

    assert p.current_state == "boiling", (
        f"持续沸腾应判定为 boiling, 实际为 {p.current_state}"
    )
    print("\n✅ 持续沸腾测试通过!")
    return True


def test_no_container():
    """
    模拟无容器场景，验证不会误报。
    """
    print("\n" + "=" * 50)
    print("端到端测试: 无容器场景")
    print("=" * 50)

    p = BoilingWaterProcessor(
        steam_threshold=10.0,
        bubble_threshold=20.0,
    )

    # 无 ROI（模拟未检测到容器）
    for i in range(5):
        steam = p.analyze_steam(np.zeros((0, 0, 3), dtype=np.uint8), None)
        bubble = p.analyze_bubbles(np.zeros((0, 0, 3), dtype=np.uint8))
        state = p.determine_state(steam, bubble)
        p.update_state(state, 0)

        print(f"帧{i:2d}: steam={steam:6.2f}, bubble={bubble:6.2f}, state={p.current_state}")

    assert p.current_state == "idle", "无容器时应保持 idle"
    assert len(p.state_history) == 0, "无容器时不应产生状态变化"
    print("\n✅ 无容器测试通过!")
    return True


if __name__ == "__main__":
    test_idle_to_boiling_sequence()
    test_constant_boiling()
    test_no_container()
    print("\n" + "=" * 50)
    print("所有端到端测试通过!")
    print("=" * 50)
