from __future__ import annotations

from dataclasses import dataclass, field

from .dto import GuiEvent, GuiState


@dataclass(slots=True)
class GuiStateMachine:
    current_state: GuiState = GuiState.IDLE
    _transitions: dict[GuiState, dict[GuiEvent, GuiState]] = field(
        default_factory=lambda: {
            GuiState.IDLE: {
                GuiEvent.START_REALTIME: GuiState.REALTIME,
                GuiEvent.IMPORT_OFFLINE: GuiState.OFFLINE,
                GuiEvent.RESET_IDLE: GuiState.IDLE,
                GuiEvent.UPDATE_OPTIONS: GuiState.IDLE,
                GuiEvent.SAVE_SNAPSHOT: GuiState.IDLE,
            },
            GuiState.REALTIME: {
                GuiEvent.STOP_REALTIME: GuiState.IDLE,
                GuiEvent.RESET_IDLE: GuiState.IDLE,
                GuiEvent.UPDATE_OPTIONS: GuiState.REALTIME,
                GuiEvent.SAVE_SNAPSHOT: GuiState.REALTIME,
            },
            GuiState.OFFLINE: {
                GuiEvent.START_REALTIME: GuiState.REALTIME,
                GuiEvent.IMPORT_OFFLINE: GuiState.OFFLINE,
                GuiEvent.RESET_IDLE: GuiState.IDLE,
                GuiEvent.UPDATE_OPTIONS: GuiState.OFFLINE,
                GuiEvent.SAVE_SNAPSHOT: GuiState.OFFLINE,
            },
        }
    )

    def target_for(self, event: GuiEvent) -> GuiState:
        target = self._transitions.get(self.current_state, {}).get(event)
        if target is None:
            raise ValueError(
                f"illegal transition: state={self.current_state.value}, event={event.value}"
            )
        return target

    def commit(self, state: GuiState) -> None:
        self.current_state = state
