"""
单元测试：验证核心算法逻辑

运行方式：
    cd vision_agent_demo/references
    python3 -m pytest tests/test_unit.py -v
    或：python3 tests/test_unit.py
"""

import os
import sys
import time
import unittest

# 将父目录加入路径，确保能导入 boiling_water_processor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from boiling_water_processor import BoilingWaterProcessor


class TestStateDetermination(unittest.TestCase):
    """测试状态判定逻辑"""

    def setUp(self):
        self.p = BoilingWaterProcessor(
            steam_threshold=10.0,
            bubble_threshold=20.0,
        )

    def test_idle_low_scores(self):
        """低分应判定为 idle"""
        self.assertEqual(self.p.determine_state(0, 0), "idle")
        self.assertEqual(self.p.determine_state(2, 5), "idle")
        self.assertEqual(self.p.determine_state(3, 7), "idle")

    def test_boiling_high_scores(self):
        """高分应判定为 boiling"""
        self.assertEqual(self.p.determine_state(15, 30), "boiling")
        self.assertEqual(self.p.determine_state(50, 50), "boiling")

    def test_heating_moderate_scores(self):
        """中等分数应判定为 heating"""
        # 蒸汽达到阈值30% (3.0) 或 气泡达到阈值40% (8.0)
        self.assertEqual(self.p.determine_state(5, 5), "heating")   # 蒸汽 > 3
        self.assertEqual(self.p.determine_state(2, 10), "heating")  # 气泡 > 8
        self.assertEqual(self.p.determine_state(8, 15), "heating")  # 都接近但未达 boiling

    def test_boundary_exact_threshold(self):
        """边界值：刚好等于阈值时"""
        # steam=10, bubble=20 应判定为 boiling（严格大于）
        self.assertEqual(self.p.determine_state(10.1, 20.1), "boiling")
        # steam=3, bubble=8 应判定为 heating（刚好超过 30%/40%）
        self.assertEqual(self.p.determine_state(3.1, 5), "heating")
        self.assertEqual(self.p.determine_state(2, 8.1), "heating")


class TestSteamDetection(unittest.TestCase):
    """测试蒸汽检测算法"""

    def setUp(self):
        self.p = BoilingWaterProcessor()

    def test_identical_frames(self):
        """相同帧应返回 0"""
        roi = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = self.p.analyze_steam(roi, roi)
        self.assertEqual(score, 0.0)

    def test_no_prev_roi(self):
        """无历史帧时应基于亮度返回分数"""
        roi = np.full((100, 100, 3), 200, dtype=np.uint8)
        score = self.p.analyze_steam(roi, None)
        self.assertGreater(score, 0)

    def test_bright_region_detected(self):
        """亮白色区域应产生明显蒸汽分"""
        roi1 = np.full((100, 100, 3), 100, dtype=np.uint8)
        roi2 = roi1.copy()
        roi2[20:80, 20:80] = [220, 220, 220]  # 亮白色区域 36%
        score = self.p.analyze_steam(roi2, roi1)
        self.assertGreater(score, 10)

    def test_motion_bonus(self):
        """运动应增加蒸汽分数"""
        base = np.full((100, 100, 3), 100, dtype=np.uint8)
        base[20:80, 20:80] = [220, 220, 220]

        moved = base.copy()
        moved[25:85, 25:85] = [230, 230, 230]  # 区域移动

        score_static = self.p.analyze_steam(base, base)
        score_moving = self.p.analyze_steam(moved, base)
        self.assertGreater(score_moving, score_static)


class TestBubbleDetection(unittest.TestCase):
    """测试气泡检测算法"""

    def setUp(self):
        self.p = BoilingWaterProcessor()

    def test_smooth_surface(self):
        """平滑水面应返回低分"""
        roi = np.full((100, 100, 3), 128, dtype=np.uint8)
        score = self.p.analyze_bubbles(roi)
        self.assertLess(score, 5)

    def test_noisy_surface(self):
        """嘈杂/气泡区域应返回高分"""
        np.random.seed(42)
        roi = np.random.randint(100, 156, (100, 100, 3), dtype=np.uint8)
        score = self.p.analyze_bubbles(roi)
        self.assertGreater(score, 10)

    def test_texture_comparison(self):
        """气泡区域纹理分应高于平静水面"""
        smooth = np.full((100, 100, 3), 128, dtype=np.uint8)
        np.random.seed(42)
        noisy = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertGreater(
            self.p.analyze_bubbles(noisy),
            self.p.analyze_bubbles(smooth),
        )


class TestStateMachine(unittest.TestCase):
    """测试状态机防抖逻辑"""

    def test_initial_state(self):
        """初始状态应为 idle"""
        p = BoilingWaterProcessor()
        self.assertEqual(p.current_state, "idle")
        self.assertEqual(len(p.state_history), 0)

    def test_debounce_prevents_quick_change(self):
        """防抖应阻止快速状态切换"""
        p = BoilingWaterProcessor(state_change_min_duration=1.0)
        p.update_state("heating", 0.8)
        self.assertEqual(p.current_state, "idle")  # 被防抖阻止

    def test_state_change_after_debounce(self):
        """超过防抖时间后应允许切换"""
        p = BoilingWaterProcessor(state_change_min_duration=0.1)
        time.sleep(0.15)
        p.update_state("heating", 0.8)
        self.assertEqual(p.current_state, "heating")
        self.assertEqual(len(p.state_history), 1)

    def test_same_state_ignored(self):
        """相同状态应被忽略"""
        p = BoilingWaterProcessor(state_change_min_duration=0.1)
        time.sleep(0.15)
        p.update_state("heating", 0.8)
        p.update_state("heating", 0.9)  # 再次更新相同状态
        self.assertEqual(len(p.state_history), 1)

    def test_multiple_transitions(self):
        """多次状态切换应正确记录"""
        p = BoilingWaterProcessor(state_change_min_duration=0.1)

        time.sleep(0.15)
        p.update_state("heating", 0.8)
        time.sleep(0.15)
        p.update_state("boiling", 0.95)

        self.assertEqual(p.current_state, "boiling")
        self.assertEqual(len(p.state_history), 2)
        self.assertEqual(p.state_history[0]["state"], "heating")
        self.assertEqual(p.state_history[1]["state"], "boiling")

    def test_history_format(self):
        """历史记录格式化应正常工作"""
        p = BoilingWaterProcessor(state_change_min_duration=0.1)
        time.sleep(0.15)
        p.update_state("heating", 0.8)

        formatted = p._format_history()
        self.assertIn("heating", formatted)
        self.assertIn("置信度", formatted)


class TestEventClasses(unittest.TestCase):
    """测试事件类定义"""

    def test_water_state_changed_event(self):
        """状态变化事件应正确创建"""
        from boiling_water_processor import WaterStateChangedEvent

        event = WaterStateChangedEvent(
            old_state="idle",
            new_state="heating",
            timestamp=1234567890.0,
            confidence=0.85,
        )
        self.assertEqual(event.type, "boiling_water.state_changed")
        self.assertEqual(event.old_state, "idle")
        self.assertEqual(event.new_state, "heating")

    def test_boiling_detected_event(self):
        """沸腾检测事件应正确创建"""
        from boiling_water_processor import BoilingDetectedEvent

        event = BoilingDetectedEvent(
            timestamp=1234567890.0,
            duration_seconds=45.0,
        )
        self.assertEqual(event.type, "boiling_water.boiling_detected")
        self.assertEqual(event.duration_seconds, 45.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
