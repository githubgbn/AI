"""
烧水状态识别处理器 (Boiling Water Processor)

基于 Vision-Agents 框架思想实现，兼容两种运行模式：
1. 独立运行：直接分析本地视频文件或摄像头
2. Agent 集成：作为 Vision-Agents 的 VideoProcessorPublisher 接入实时流

核心能力：
- YOLO 目标检测定位水壶/容器
- 帧间差分 + 纹理分析检测蒸汽和气泡
- 状态机管理烧水状态（idle / heating / boiling）
- 事件驱动，支持与 Agent 联动
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# 尝试导入 Vision-Agents 组件（如果已安装）
try:
    from vision_agents.core.processors.base_processor import VideoProcessorPublisher
    from vision_agents.core.events.base import PluginBaseEvent
    from vision_agents.core.events.manager import EventManager
    from vision_agents.core.warmup import Warmable

    VISION_AGENTS_AVAILABLE = True
except ImportError:
    VISION_AGENTS_AVAILABLE = False

    # 定义基类占位符，使代码在无 Vision-Agents 环境时也能独立运行
    class PluginBaseEvent:
        type: str = ""

    class VideoProcessorPublisher:
        name: str = ""

        async def close(self) -> None:
            pass

    class Warmable:
        async def on_warmup(self) -> Any:
            return None

        def on_warmed_up(self, resource: Any) -> None:
            pass


@dataclass
class WaterStateChangedEvent(PluginBaseEvent):
    """烧水状态变化事件"""

    type: str = field(default="boiling_water.state_changed", init=False)
    old_state: str = ""
    new_state: str = ""
    timestamp: float = 0.0
    confidence: float = 0.0


@dataclass
class BoilingDetectedEvent(PluginBaseEvent):
    """沸腾检测事件"""

    type: str = field(default="boiling_water.boiling_detected", init=False)
    timestamp: float = 0.0
    duration_seconds: float = 0.0


class BoilingWaterProcessor(VideoProcessorPublisher, Warmable):
    """
    烧水状态识别处理器

    检测流程：
    1. YOLO 检测画面中的水壶/锅/容器
    2. 在容器 ROI 及上方区域分析视觉特征：
       - 蒸汽：帧间差分检测上升的白色/灰色雾气
       - 气泡：水面纹理复杂度变化
    3. 综合特征分数，通过状态机判定当前烧水状态
    4. 输出带状态标注的视频帧，并通过事件通知状态变化

    Args:
        fps: 帧处理率（默认 5）
        yolo_model_path: YOLO 模型路径（默认 yolo11n.pt）
        device: 运行设备（默认 cpu）
        steam_threshold: 蒸汽检测判定阈值（默认 15.0）
        bubble_threshold: 气泡活跃度判定阈值（默认 30.0）
        state_change_min_duration: 状态保持最短时间（秒），防止抖动（默认 3.0）
    """

    name = "boiling_water"

    # 状态常量
    STATE_IDLE = "idle"  # 未加热
    STATE_HEATING = "heating"  # 加热中
    STATE_BOILING = "boiling"  # 沸腾

    def __init__(
        self,
        fps: int = 5,
        yolo_model_path: str = "yolo11n.pt",
        device: str = "cpu",
        steam_threshold: float = 15.0,
        bubble_threshold: float = 30.0,
        state_change_min_duration: float = 3.0,
    ):
        self.fps = fps
        self.yolo_model_path = yolo_model_path
        self.device = device
        self.steam_threshold = steam_threshold
        self.bubble_threshold = bubble_threshold
        self.state_change_min_duration = state_change_min_duration

        # YOLO 模型（懒加载）
        self.yolo_model = None

        # 状态管理
        self.current_state = self.STATE_IDLE
        self.state_start_time = time.time()
        self.state_history: List[Dict[str, Any]] = []

        # 帧缓存（用于帧间差分）
        self.prev_roi = None

        # 事件管理器
        self.events = EventManager() if VISION_AGENTS_AVAILABLE else _DummyEventManager()
        self.events.register(WaterStateChangedEvent)
        self.events.register(BoilingDetectedEvent)

        self.logger = logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # 模型加载
    # ------------------------------------------------------------------

    async def on_warmup(self) -> Any:
        """预加载 YOLO 模型。Vision-Agents 会在 Agent 启动时自动调用此方法。"""
        try:
            from ultralytics import YOLO

            loop = asyncio.get_event_loop()

            def _load():
                model = YOLO(self.yolo_model_path)
                model.to(self.device)
                return model

            self.yolo_model = await loop.run_in_executor(None, _load)
            self.logger.info(f"YOLO 模型加载成功: {self.yolo_model_path}")
            return self.yolo_model
        except Exception as e:
            self.logger.warning(f"YOLO 模型加载失败: {e}")
            return None

    def on_warmed_up(self, resource: Any) -> None:
        """Vision-Agents 回调：模型加载完成后调用。"""
        if resource is not None:
            self.yolo_model = resource

    def _ensure_model(self):
        """确保模型已加载（独立运行模式用）。"""
        if self.yolo_model is None:
            try:
                from ultralytics import YOLO

                self.yolo_model = YOLO(self.yolo_model_path)
                self.yolo_model.to(self.device)
                self.logger.info(f"YOLO 模型加载成功: {self.yolo_model_path}")
            except Exception as e:
                self.logger.error(f"YOLO 模型加载失败: {e}")

    # ------------------------------------------------------------------
    # 目标检测
    # ------------------------------------------------------------------

    def detect_container(self, frame: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """
        使用 YOLO 检测画面中的容器（水壶、锅、杯子等）。

        Returns:
            (x1, y1, x2, y2) 检测框坐标，未检测到则返回 None
        """
        if self.yolo_model is None:
            return None

        results = self.yolo_model(frame, verbose=False)
        best_box = None
        best_conf = 0.0

        # 容器相关类别关键词
        container_keywords = ["bottle", "cup", "bowl", "vase", "pot", "kettle", "mug"]

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                name = result.names.get(cls_id, "").lower()

                if any(kw in name for kw in container_keywords) and conf > best_conf:
                    best_conf = conf
                    xyxy = box.xyxy[0].cpu().numpy().astype(int)
                    best_box = (int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3]))

        return best_box

    # ------------------------------------------------------------------
    # 视觉特征分析
    # ------------------------------------------------------------------

    def analyze_steam(self, roi: np.ndarray, prev_roi: Optional[np.ndarray]) -> float:
        """
        分析 ROI 区域内的蒸汽特征。

        原理：蒸汽通常表现为亮度较高、颜色偏白/灰的区域。
        检测策略以颜色/亮度特征为主（检测蒸汽"存在"），
        帧间差分为辅（检测蒸汽"运动"），避免蒸汽稳定后因差分为零而漏检。

        Args:
            roi: 当前帧 ROI
            prev_roi: 上一帧 ROI（用于帧间差分，辅助加权）

        Returns:
            蒸汽指数（0-100），数值越大表示蒸汽越明显
        """
        total_pixels = roi.shape[0] * roi.shape[1]
        if total_pixels == 0:
            return 0.0

        # 主检测：亮白色/浅灰色区域（蒸汽颜色特征）
        # 蒸汽通常是半透明的白色/灰色，降低阈值以捕获更淡的蒸汽
        bright_mask = cv2.inRange(roi, np.array([150, 150, 150]), np.array([255, 255, 255]))
        bright_ratio = np.sum(bright_mask > 0) / total_pixels * 100

        # 辅助：帧间差分检测运动（蒸汽在飘动）
        motion_bonus = 0.0
        if prev_roi is not None and roi.shape == prev_roi.shape:
            diff = cv2.absdiff(
                cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev_roi, cv2.COLOR_BGR2GRAY),
            )
            _, motion_mask = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
            motion_ratio = np.sum(motion_mask > 0) / total_pixels * 100
            motion_bonus = motion_ratio * 0.5  # 运动作为加分项

        # 蒸汽分数 = 亮度基础分 + 运动加分
        steam_score = bright_ratio + motion_bonus
        return min(steam_score, 100.0)

    def analyze_bubbles(self, roi: np.ndarray) -> float:
        """
        分析 ROI 区域内的气泡活跃度。

        原理：沸腾时水面产生大量密集小气泡，导致局部纹理复杂度显著增加。
        使用拉普拉斯算子提取边缘，以方差量化纹理复杂度。

        Args:
            roi: 当前帧 ROI

        Returns:
            气泡活跃度指数（0-100）
        """
        if roi is None or roi.size == 0:
            return 0.0

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 拉普拉斯边缘检测
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = np.var(laplacian)

        # 归一化到 0-100（经验系数，可调整）
        normalized = min(texture_variance / 50.0, 100.0)

        return normalized

    # ------------------------------------------------------------------
    # 状态判定
    # ------------------------------------------------------------------

    def determine_state(self, steam_score: float, bubble_score: float) -> str:
        """
        根据蒸汽指数和气泡活跃度判定烧水状态。

        判定规则：
        - boiling: 蒸汽明显（> 阈值）且 气泡活跃（> 阈值）
        - heating: 蒸汽开始出现（> 30% 阈值）或 气泡开始增多（> 40% 阈值）
        - idle: 以上都不满足
        """
        if steam_score > self.steam_threshold and bubble_score > self.bubble_threshold:
            return self.STATE_BOILING
        elif steam_score > self.steam_threshold * 0.3 or bubble_score > self.bubble_threshold * 0.4:
            return self.STATE_HEATING
        else:
            return self.STATE_IDLE

    def update_state(self, new_state: str, confidence: float = 1.0, timestamp: Optional[float] = None):
        """
        更新烧水状态，满足最小持续时间要求后触发事件。

        Args:
            new_state: 新状态
            confidence: 置信度（0-1）
            timestamp: 可选的外部时间戳（用于视频文件处理等非实时场景）
        """
        if new_state == self.current_state:
            return

        now = timestamp if timestamp is not None else time.time()
        state_duration = now - self.state_start_time

        # 防抖：状态持续时间太短则忽略
        if state_duration < self.state_change_min_duration:
            return

        old_state = self.current_state
        self.current_state = new_state
        self.state_start_time = now

        # 记录历史
        self.state_history.append(
            {
                "state": new_state,
                "timestamp": now,
                "duration": state_duration,
                "confidence": confidence,
            }
        )

        self.logger.info(
            f"【状态变化】{old_state} -> {new_state} (持续 {state_duration:.1f}s, 置信度 {confidence:.2f})"
        )

        # 触发事件（兼容同步和异步环境）
        try:
            asyncio.create_task(
                self.events.emit(
                    WaterStateChangedEvent(
                        old_state=old_state,
                        new_state=new_state,
                        timestamp=now,
                        confidence=confidence,
                    )
                )
            )

            if new_state == self.STATE_BOILING:
                asyncio.create_task(
                    self.events.emit(
                        BoilingDetectedEvent(
                            timestamp=now,
                            duration_seconds=state_duration,
                        )
                    )
                )
        except RuntimeError:
            # 无运行中事件循环时跳过异步事件触发（本地同步模式）
            pass

    # ------------------------------------------------------------------
    # 视频标注
    # ------------------------------------------------------------------

    def annotate_frame(
        self,
        frame: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]],
        state: str,
        steam_score: float,
        bubble_score: float,
    ) -> np.ndarray:
        """
        在视频帧上绘制状态信息面板和检测框。

        Returns:
            标注后的帧
        """
        annotated = frame.copy()

        # 状态颜色映射
        colors = {
            self.STATE_IDLE: (0, 255, 0),  # 绿色
            self.STATE_HEATING: (0, 165, 255),  # 橙色
            self.STATE_BOILING: (0, 0, 255),  # 红色
        }
        color = colors.get(state, (128, 128, 128))

        # 绘制检测框
        if bbox:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                annotated,
                "Container",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )

        # 信息面板
        px, py = 10, 10
        pw, ph = 300, 130
        cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (0, 0, 0), -1)
        cv2.rectangle(annotated, (px, py), (px + pw, py + ph), (255, 255, 255), 1)

        state_labels = {
            self.STATE_IDLE: "状态: 未加热 (IDLE)",
            self.STATE_HEATING: "状态: 加热中 (HEATING)",
            self.STATE_BOILING: "状态: 沸腾 (BOILING!)",
        }

        lines = [
            state_labels.get(state, f"状态: {state}"),
            f"蒸汽指数: {steam_score:.1f}",
            f"气泡指数: {bubble_score:.1f}",
            f"状态持续: {time.time() - self.state_start_time:.1f}s",
            f"历史变化: {len(self.state_history)} 次",
        ]

        for i, line in enumerate(lines):
            y = py + 25 + i * 22
            cv2.putText(
                annotated,
                line,
                (px + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255) if i == 0 else (200, 200, 200),
                2 if i == 0 else 1,
            )

        return annotated

    # ------------------------------------------------------------------
    # 独立运行模式：本地视频/摄像头
    # ------------------------------------------------------------------

    async def process_local_video(self, video_source: Any):
        """
        处理本地视频文件或摄像头输入（独立运行模式，无需 Stream API）。

        Args:
            video_source: 视频文件路径（str）或摄像头索引（int，如 0）
        """
        # 打开视频源
        if isinstance(video_source, str):
            cap = cv2.VideoCapture(video_source)
            source_name = video_source
        else:
            cap = cv2.VideoCapture(int(video_source))
            source_name = f"camera:{video_source}"

        if not cap.isOpened():
            self.logger.error(f"无法打开视频源: {source_name}")
            return

        # 确保模型已加载
        self._ensure_model()

        fps = cap.get(cv2.CAP_PROP_FPS) or self.fps
        delay = max(1, int(1000 / fps))

        self.logger.info(f"开始处理视频源: {source_name} (fps: {fps:.1f})")
        self.logger.info("按 'q' 键退出")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("视频读取结束")
                    break

                # 1. 检测容器
                bbox = self.detect_container(frame)

                # 2. 提取 ROI（包含容器及上方蒸汽区域）
                roi = None
                if bbox:
                    x1, y1, x2, y2 = bbox
                    # 向上扩展 ROI 以捕获蒸汽
                    margin = int((y2 - y1) * 0.6)
                    roi_y1 = max(0, y1 - margin)
                    roi = frame[roi_y1:y2, x1:x2]

                # 3. 分析视觉特征
                steam_score = 0.0
                bubble_score = 0.0

                if roi is not None and roi.size > 0:
                    steam_score = self.analyze_steam(roi, self.prev_roi)
                    bubble_score = self.analyze_bubbles(roi)
                    self.prev_roi = roi.copy()

                # 4. 状态判定与更新
                new_state = self.determine_state(steam_score, bubble_score)
                confidence = min(1.0, (steam_score + bubble_score) / 100.0)
                self.update_state(new_state, confidence)

                # 5. 标注并显示
                annotated = self.annotate_frame(
                    frame, bbox, self.current_state, steam_score, bubble_score
                )
                cv2.imshow("Boiling Water Detection", annotated)

                if cv2.waitKey(delay) & 0xFF == ord("q"):
                    self.logger.info("用户主动退出")
                    break

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.logger.info("视频处理已结束")
            self.logger.info(f"完整状态历史:\n{self._format_history()}")

    def _format_history(self) -> str:
        """格式化状态历史为可读字符串。"""
        if not self.state_history:
            return "  无状态变化"
        lines = []
        for i, entry in enumerate(self.state_history, 1):
            ts = time.strftime("%H:%M:%S", time.localtime(entry["timestamp"]))
            lines.append(
                f"  [{i}] {ts} | {entry['state']:8s} | "
                f"持续 {entry['duration']:.1f}s | 置信度 {entry['confidence']:.2f}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Vision-Agents 兼容接口（实时流模式）
    # ------------------------------------------------------------------

    async def process_video(self, track, participant_id=None, shared_forwarder=None):
        """
        Vision-Agents 实时流处理接口。

        当 Agent 接收到远程参与者的视频轨道时，会自动调用此方法。
        处理逻辑与本地模式一致，只是视频源从文件/摄像头变为 WebRTC 轨道。
        """
        # 注：完整实现需要接入 aiortc 的视频帧读取逻辑
        # 此处提供框架，实际使用时根据 Vision-Agents 版本调整
        self.logger.info(f"开始处理参与者视频: {participant_id}")
        # 实际实现参考 Vision-Agents 官方文档的 VideoProcessor 接入方式

    def publish_video_track(self):
        """发布处理后的视频轨道（供远程参与者观看标注后的画面）。"""
        # 实际实现需要创建 aiortc.VideoStreamTrack 子类
        pass

    async def close(self) -> None:
        """清理资源。"""
        self.prev_roi = None
        self.logger.info("Processor 资源已清理")


# ------------------------------------------------------------------
# 事件管理器占位符（独立运行模式用）
# ------------------------------------------------------------------

class _DummyEventManager:
    """当 Vision-Agents 未安装时的事件管理器占位符，保持接口一致。"""

    def register(self, event_class):
        pass

    def emit(self, event):
        # 独立模式下直接打印事件信息
        logging.getLogger(__name__).info(f"[事件] {event.type}: {event}")


# ------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    processor = BoilingWaterProcessor(
        fps=5,
        yolo_model_path="yolo11n.pt",
        steam_threshold=15.0,
        bubble_threshold=30.0,
    )

    # 解析命令行参数
    source = sys.argv[1] if len(sys.argv) > 1 else "0"
    # 如果是数字则视为摄像头索引，否则视为文件路径
    try:
        source = int(source)
    except ValueError:
        pass

    asyncio.run(processor.process_local_video(source))
