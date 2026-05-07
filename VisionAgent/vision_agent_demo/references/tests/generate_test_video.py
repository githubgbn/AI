"""
生成模拟烧水测试视频

运行方式：
    python3 tests/generate_test_video.py

输出：
    test_boiling.mp4 —— 模拟从静止到沸腾的 12 秒测试视频

视频内容：
    0-4s:   平静水面（idle）
    4-8s:   出现少量蒸汽和小气泡（heating）
    8-12s:  大量蒸汽和剧烈气泡（boiling）
"""

import os
import random

import cv2
import numpy as np

# 视频参数
FPS = 5
WIDTH, HEIGHT = 640, 480
DURATION_SECONDS = 12
TOTAL_FRAMES = FPS * DURATION_SECONDS

# 阶段划分
PHASE_IDLE = (0, 4)      # 0-4 秒
PHASE_HEATING = (4, 8)   # 4-8 秒
PHASE_BOILING = (8, 12)  # 8-12 秒

# 输出路径
OUTPUT_PATH = "test_boiling.mp4"


def draw_container(frame):
    """在画面中绘制水壶/锅。"""
    # 壶身
    cv2.rectangle(frame, (240, 280), (400, 420), (80, 60, 40), -1)
    cv2.rectangle(frame, (240, 280), (400, 420), (40, 30, 20), 2)
    # 壶嘴
    cv2.fillPoly(frame, [np.array([[400, 320], [460, 300], [460, 330], [400, 350]])], (80, 60, 40))
    # 壶把手
    cv2.ellipse(frame, (220, 350), (30, 60), 0, 0, 360, (60, 40, 20), 6)
    # 壶盖
    cv2.ellipse(frame, (320, 280), (60, 15), 0, 0, 360, (100, 80, 60), -1)
    cv2.ellipse(frame, (320, 280), (60, 15), 0, 0, 360, (50, 40, 30), 2)


def add_steam(frame, intensity, seed_offset):
    """在画面上方添加蒸汽效果。"""
    random.seed(42 + seed_offset)
    h, w = frame.shape[:2]

    # 蒸汽粒子数量与强度成正比
    num_particles = int(intensity * 50)
    for _ in range(num_particles):
        # 蒸汽从壶嘴和壶盖上方升起
        cx = random.randint(280, 420)
        cy = random.randint(140, 260)
        radius = random.randint(8, 30)
        alpha = random.uniform(0.15, 0.55) * intensity

        overlay = frame.copy()
        color = (245, 245, 245)
        cv2.circle(overlay, (cx, cy), radius, color, -1)
        frame = cv2.addWeighted(frame, 1 - alpha, overlay, alpha, 0)

    return frame


def add_bubbles(frame, intensity, seed_offset):
    """在水面区域添加气泡效果。"""
    random.seed(100 + seed_offset)

    num_bubbles = int(intensity * 40)
    for _ in range(num_bubbles):
        cx = random.randint(250, 390)
        cy = random.randint(340, 415)
        radius = random.randint(2, 5)
        brightness = random.randint(200, 255)
        cv2.circle(frame, (cx, cy), radius, (brightness, brightness, brightness), -1)

    return frame


def generate_frame(frame_idx, total_frames):
    """生成单帧画面。"""
    frame = np.full((HEIGHT, WIDTH, 3), (60, 80, 100), dtype=np.uint8)  # 深蓝灰背景

    # 绘制桌面
    cv2.rectangle(frame, (0, 420), (WIDTH, HEIGHT), (120, 100, 80), -1)

    # 绘制水壶
    draw_container(frame)

    # 计算当前时间
    time_sec = frame_idx / FPS

    # 根据阶段添加效果
    if PHASE_IDLE[0] <= time_sec < PHASE_IDLE[1]:
        # idle: 无蒸汽无气泡
        pass

    elif PHASE_HEATING[0] <= time_sec < PHASE_HEATING[1]:
        # heating: 少量蒸汽和气泡
        progress = (time_sec - PHASE_HEATING[0]) / (PHASE_HEATING[1] - PHASE_HEATING[0])
        intensity = progress * 0.5  # 逐渐增强到 0.5
        frame = add_steam(frame, intensity, frame_idx)
        frame = add_bubbles(frame, intensity, frame_idx)

    elif PHASE_BOILING[0] <= time_sec < PHASE_BOILING[1]:
        # boiling: 大量蒸汽和剧烈气泡
        progress = (time_sec - PHASE_BOILING[0]) / (PHASE_BOILING[1] - PHASE_BOILING[0])
        intensity = 0.5 + progress * 0.5  # 从 0.5 增强到 1.0
        frame = add_steam(frame, intensity, frame_idx)
        frame = add_bubbles(frame, intensity, frame_idx)

    return frame


def main():
    print("=" * 50)
    print("生成模拟烧水测试视频")
    print("=" * 50)
    print(f"分辨率: {WIDTH}x{HEIGHT}")
    print(f"帧率: {FPS} fps")
    print(f"时长: {DURATION_SECONDS} 秒")
    print(f"总帧数: {TOTAL_FRAMES}")
    print(f"输出: {OUTPUT_PATH}")
    print("=" * 50)

    # 创建视频写入器
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (WIDTH, HEIGHT))

    if not writer.isOpened():
        print("错误: 无法创建视频文件，尝试 AVI 格式...")
        OUTPUT_PATH_AVI = OUTPUT_PATH.replace(".mp4", ".avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(OUTPUT_PATH_AVI, fourcc, FPS, (WIDTH, HEIGHT))
        if not writer.isOpened():
            raise RuntimeError("无法创建视频文件，请检查 OpenCV 视频编码支持")
        print(f"使用 AVI 格式: {OUTPUT_PATH_AVI}")

    # 逐帧生成
    for i in range(TOTAL_FRAMES):
        frame = generate_frame(i, TOTAL_FRAMES)
        writer.write(frame)

        # 每 1 秒打印进度
        if i % FPS == 0:
            print(f"生成中... {i // FPS}s / {DURATION_SECONDS}s")

    writer.release()
    print("=" * 50)
    print(f"✅ 视频生成完成: {OUTPUT_PATH}")
    print("=" * 50)
    print("\n运行 demo 验证：")
    print(f"  python3 demo_local.py {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
