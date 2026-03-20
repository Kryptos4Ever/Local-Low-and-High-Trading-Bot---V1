"""state/ — Persistencia de estado entre sesiones."""
from .state_manager import (
    Checkpoint, StateManager,
    MemoryStateManager, JSONStateManager,
    restore_wallet_from_checkpoint, build_state_manager,
)

__all__ = [
    "Checkpoint", "StateManager",
    "MemoryStateManager", "JSONStateManager",
    "restore_wallet_from_checkpoint", "build_state_manager",
]
