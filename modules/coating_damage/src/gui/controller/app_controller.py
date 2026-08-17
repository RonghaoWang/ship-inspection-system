from __future__ import annotations

from typing import Any

from .dto import (
    GuiEvent,
    GuiState,
    OfflineService,
    RealtimeService,
    SaveService,
    ServiceResult,
)
from .state_machine import GuiStateMachine
from ..services import OfflineImportService, RealtimeCameraService, SnapshotSaveService


class AppController:
    def __init__(
        self,
        view: Any,
        machine: GuiStateMachine | None = None,
        realtime_service: RealtimeService | None = None,
        offline_service: OfflineService | None = None,
        save_service: SaveService | None = None,
    ) -> None:
        self._view = view
        self._machine = machine or GuiStateMachine()
        self._realtime_service = realtime_service or RealtimeCameraService()
        self._offline_service = offline_service or OfflineImportService()
        self._save_service = save_service or SnapshotSaveService()

        collect_options = getattr(self._view, "collect_runtime_options", None)
        self._latest_options: dict[str, Any] = (
            dict(collect_options()) if callable(collect_options) else {}
        )
        self._latest_frame: object | None = None

        self._bind_view_events()
        self._bind_service_callbacks()
        self._view.render_state(self._machine.current_state.value)

    @property
    def current_state(self) -> GuiState:
        return self._machine.current_state

    def _bind_view_events(self) -> None:
        self._view.start_requested.connect(self.on_start_realtime)
        self._view.stop_requested.connect(self.on_stop_realtime)
        self._view.import_requested.connect(self.on_import_offline)
        self._view.save_requested.connect(self.on_save_snapshot)
        self._view.options_changed.connect(self.on_update_options)

    def _bind_service_callbacks(self) -> None:
        self._realtime_service.set_callbacks(
            on_frame=self._on_frame,
            on_status=self._on_status,
            on_error=self._on_error,
        )
        self._offline_service.set_callbacks(
            on_frame=self._on_frame,
            on_status=self._on_status,
            on_error=self._on_error,
        )

    def _on_frame(self, frame: object) -> None:
        self._latest_frame = frame
        self._save_service.update_latest_frame(frame)
        self._view.render_frame(frame)

    def _on_status(self, text: str) -> None:
        self._view.render_status(text)

    def _fallback_to_idle(self) -> None:
        if self._machine.current_state == GuiState.IDLE:
            return
        self._machine.commit(GuiState.IDLE)
        self._clear_display_cache()
        self._view.render_state(GuiState.IDLE.value)

    def _on_error(self, detail: str) -> None:
        self._view.show_error(detail)
        self._fallback_to_idle()

    def _handle_service_result(self, result: ServiceResult) -> bool:
        if result.message:
            if result.success:
                self._view.render_status(result.message)
            else:
                self._view.show_error(result.message)
        return bool(result.success)

    def _clear_display_cache(self) -> None:
        self._latest_frame = None
        self._view.clear_display()

    def _commit_state(self, state: GuiState) -> None:
        self._machine.commit(state)
        self._view.render_state(state.value)

    def _leave_state(self, state: GuiState) -> bool:
        if state == GuiState.REALTIME:
            return self._handle_service_result(self._realtime_service.stop_realtime())
        elif state == GuiState.OFFLINE:
            self._offline_service.clear_context()
        return True

    def _enter_state(self, state: GuiState, payload: Any = None) -> bool:
        if state == GuiState.REALTIME:
            return self._handle_service_result(
                self._realtime_service.start_realtime(payload)
            )
        if state == GuiState.OFFLINE:
            return self._handle_service_result(
                self._offline_service.run_offline(payload)
            )
        return True

    def _handle_state_change(
        self, source: GuiState, target: GuiState, payload: Any = None
    ) -> bool:
        if not self._leave_state(source):
            return False

        # 旧状态的资源在此之后已消耗完，因此失败的进入操作无法安全地回滚到源状态。回退到闲置状态
        self._clear_display_cache()
        if self._enter_state(target, payload):
            self._commit_state(target)
            return True

        self._commit_state(GuiState.IDLE)
        return False

    def _handle_same_state_event(self, event: GuiEvent, payload: Any = None) -> bool:
        if event == GuiEvent.UPDATE_OPTIONS:
            return self._handle_update_options(payload)
        if event == GuiEvent.SAVE_SNAPSHOT:
            return self._handle_service_result(
                self._save_service.save_snapshot(payload)
            )
        if event == GuiEvent.IMPORT_OFFLINE:
            return self._handle_service_result(
                self._offline_service.run_offline(payload)
            )
        return True

    def _transition(self, event: GuiEvent, payload: Any = None) -> bool:
        source = self._machine.current_state
        try:
            target = self._machine.target_for(event)
        except ValueError as exc:
            self._view.show_error(str(exc))
            return False

        if source != target:
            return self._handle_state_change(source, target, payload)
        return self._handle_same_state_event(event, payload)

    def _handle_update_options(self, payload: dict[str, Any] | None) -> bool:
        self._latest_options = dict(payload or {})
        state = self._machine.current_state
        if state == GuiState.REALTIME:
            return self._handle_service_result(
                self._realtime_service.update_runtime_options(self._latest_options)
            )
        if state == GuiState.OFFLINE:
            return self._handle_service_result(
                self._offline_service.reprocess_offline(self._latest_options)
            )
        # self._view.render_status("[Options] 已更新（idle）")
        return True

    def on_start_realtime(self) -> bool:
        return self._transition(GuiEvent.START_REALTIME, self._latest_options)

    def on_stop_realtime(self) -> bool:
        return self._transition(GuiEvent.STOP_REALTIME)

    def on_import_offline(self) -> bool:
        rgb_path = self._view.pick_import_rgb_path()
        if not rgb_path:
            return False
        payload = {"rgb_path": rgb_path, "options": dict(self._latest_options)}
        return self._transition(GuiEvent.IMPORT_OFFLINE, payload)

    def on_reset_idle(self) -> bool:
        return self._transition(GuiEvent.RESET_IDLE)

    def on_update_options(self, payload: dict[str, Any]) -> bool:
        return self._transition(GuiEvent.UPDATE_OPTIONS, payload)

    def on_save_snapshot(self) -> bool:
        payload = self._latest_frame
        return self._transition(GuiEvent.SAVE_SNAPSHOT, payload)
