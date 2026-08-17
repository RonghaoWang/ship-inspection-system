"""气密性检测 - GUI 适配层。

薄封装：把 leak_pipeline.LeakPipeline 暴露给 GUI。GUI 会自己起 QThread 消费生成器。
"""
from __future__ import annotations

from pathlib import Path

from modules.airtightness.leak_pipeline import (
    FrameResult,
    LeakDetectionParams,
    LeakPipeline,
    VideoSummary,
    probe_video_info,
)


class AirtightnessAdapter:
    """气密性适配器。管理一个 LeakPipeline 实例，跨视频复用。"""

    def __init__(self) -> None:
        self._pipeline = LeakPipeline()

    def check_environment(self) -> tuple[bool, str]:
        return self._pipeline.check_environment()

    def is_ready(self) -> bool:
        return self._pipeline._model is not None

    def load(self) -> None:
        self._pipeline.load()

    def process_video(self, video_path, params: LeakDetectionParams | None = None,
                      should_stop=None, writer_output_path=None):
        return self._pipeline.process_video(
            video_path, params, should_stop, writer_output_path
        )

    @staticmethod
    def probe(video_path) -> dict:
        return probe_video_info(video_path)
