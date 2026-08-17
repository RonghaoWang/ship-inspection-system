from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol


class GuiState(str, Enum):
    IDLE = "idle"
    REALTIME = "realtime"
    OFFLINE = "offline"


class GuiEvent(str, Enum):
    START_REALTIME = "start_realtime"
    STOP_REALTIME = "stop_realtime"
    IMPORT_OFFLINE = "import_offline"
    RESET_IDLE = "reset_idle"
    UPDATE_OPTIONS = "update_options"
    SAVE_SNAPSHOT = "save_snapshot"


@dataclass(slots=True)
class ServiceResult:
    success: bool
    message: str = ""
    payload: Any = None


class RealtimeService(Protocol):
    def set_callbacks(
        self,
        on_frame: Callable[[object], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None: ...

    def start_realtime(self, payload: Any = None) -> ServiceResult: ...

    def stop_realtime(self) -> ServiceResult: ...

    def update_runtime_options(self, payload: Any = None) -> ServiceResult: ...


class OfflineService(Protocol):
    def set_callbacks(
        self,
        on_frame: Callable[[object], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_error: Callable[[str], None] | None = None,
    ) -> None: ...

    def run_offline(self, payload: Any = None) -> ServiceResult: ...

    def reprocess_offline(self, payload: Any = None) -> ServiceResult: ...

    def clear_context(self) -> None: ...


class SaveService(Protocol):
    def update_latest_frame(self, frame: object) -> None: ...

    def save_snapshot(self, payload: Any = None) -> ServiceResult: ...
