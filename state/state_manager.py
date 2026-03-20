"""
state_manager.py — Persistencia de estado entre sesiones
══════════════════════════════════════════════════════════
Responsabilidad única: guardar y restaurar el estado completo del sistema
para que pueda reiniciarse sin perder contexto tras una caída o parada.

Problema que resuelve
──────────────────────
Si el programa se cae a las 2am con posiciones abiertas y se reinicia a las
3am, necesita saber:
  · Cuánto USDT y BTC tiene
  · Qué posiciones están abiertas (precio de entrada, cantidad)
  · Cuál es el slot actual y el btc_por_venta
  · Cuándo fue la última vela procesada

Sin StateManager el sistema reinicia desde cero y puede duplicar
posiciones o ignorar las que ya tenía abiertas.

Diseño
───────
  · Archivo JSON append-only: cada checkpoint es una línea nueva.
    Nunca se sobreescribe — el archivo crece. Esto garantiza que aunque
    el proceso muera a mitad de un write, el checkpoint anterior sigue
    siendo válido.

  · restore() lee el último checkpoint válido del archivo.

  · En backtest el StateManager es opcional (MemoryStateManager no escribe
    nada al disco).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional

from actors.wallet      import Wallet, MemoryWallet
from support.logger     import get_logger
from support.time_utils import now_epoch_s, to_iso

log = get_logger("state_manager")


# ══════════════════════════════════════════════════════════════════════════════
# TIPO: Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Checkpoint:
    """
    Snapshot completo del estado del sistema en un momento dado.
    Suficiente para restaurar la sesión sin pérdida de información.
    """
    ts:                    int      # epoch s UTC del checkpoint
    last_candle_ts:        int      # timestamp de la última vela procesada
    usdt_balance:          float
    btc_libre:             float
    slot_usdt:             float
    btc_por_venta:         float
    positions:             list     # [{"entry_price": float, "btc": float, "opened_at": int}]
    positions_count:       int
    portfolio_value:       float    # valorado al close de la última vela
    metadata:              Dict[str, Any]  # info extra (estrategia, config, etc.)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Checkpoint":
        return cls(**d)

    @classmethod
    def from_wallet(
        cls,
        wallet:          Wallet,
        last_candle_ts:  int,
        current_price:   float,
        metadata:        Optional[Dict[str, Any]] = None,
    ) -> "Checkpoint":
        """Construye un Checkpoint desde el estado actual de la Wallet."""
        positions = [
            {
                "entry_price": p.entry_price,
                "btc":         p.btc,
                "opened_at":   p.opened_at,
            }
            for p in wallet.get_positions()
        ]
        return cls(
            ts              = now_epoch_s(),
            last_candle_ts  = last_candle_ts,
            usdt_balance    = wallet.get_usdt_balance(),
            btc_libre       = wallet.get_btc_balance(),
            slot_usdt       = wallet.get_slot_usdt(),
            btc_por_venta   = wallet.get_btc_por_venta(),
            positions       = positions,
            positions_count = wallet.positions_count,
            portfolio_value = wallet.portfolio_value(current_price),
            metadata        = metadata or {},
        )


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ ABSTRACTA
# ══════════════════════════════════════════════════════════════════════════════

class StateManager(ABC):

    @abstractmethod
    def save(self, checkpoint: Checkpoint) -> None:
        """Persiste el checkpoint. Llamar después de cada operación ejecutada."""

    @abstractmethod
    def restore(self) -> Optional[Checkpoint]:
        """
        Retorna el último checkpoint válido, o None si no hay ninguno.
        Llamar al arrancar el sistema antes de procesar cualquier vela.
        """

    @abstractmethod
    def clear(self) -> None:
        """Borra el historial de checkpoints. Usar al iniciar un nuevo backtest."""


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN: MemoryStateManager
# ══════════════════════════════════════════════════════════════════════════════

class MemoryStateManager(StateManager):
    """
    Solo RAM — no escribe nada al disco.
    Para backtest y grid search donde no se necesita persistencia.
    """

    def __init__(self) -> None:
        self._last: Optional[Checkpoint] = None

    def save(self, checkpoint: Checkpoint) -> None:
        self._last = checkpoint

    def restore(self) -> Optional[Checkpoint]:
        return self._last

    def clear(self) -> None:
        self._last = None


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN: JSONStateManager
# ══════════════════════════════════════════════════════════════════════════════

class JSONStateManager(StateManager):
    """
    Persiste checkpoints en archivo JSONL (JSON Lines) append-only.
    Cada checkpoint es una línea JSON independiente — si el proceso muere
    a mitad de un write, las líneas anteriores siguen siendo válidas.

    restore() lee el archivo de atrás hacia adelante hasta encontrar
    el primer checkpoint válido (el más reciente).
    """

    def __init__(self, state_path: str) -> None:
        self._path = Path(state_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        log.info("JSONStateManager inicializado", path=str(self._path))

    def save(self, checkpoint: Checkpoint) -> None:
        """Append-only: agrega una línea al final del archivo."""
        line = json.dumps(checkpoint.to_dict(), default=str, ensure_ascii=False)
        with open(self._path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        log.debug(
            "checkpoint guardado",
            ts=to_iso(checkpoint.ts),
            positions=checkpoint.positions_count,
            usdt=f"{checkpoint.usdt_balance:.2f}",
        )

    def restore(self) -> Optional[Checkpoint]:
        """
        Lee el último checkpoint válido del archivo.
        Itera de atrás hacia adelante para encontrar el más reciente.
        """
        if not self._path.exists():
            log.info("sin checkpoint previo — inicio fresco")
            return None

        last_valid: Optional[Checkpoint] = None
        with open(self._path, encoding="utf-8") as f:
            lines = f.readlines()

        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                last_valid = Checkpoint.from_dict(data)
                break
            except (json.JSONDecodeError, TypeError, KeyError):
                continue   # línea corrupta — probar la anterior

        if last_valid:
            log.info(
                "checkpoint restaurado",
                ts=to_iso(last_valid.ts),
                positions=last_valid.positions_count,
                usdt=f"{last_valid.usdt_balance:.2f}",
            )
        else:
            log.warning("archivo de estado existe pero no tiene checkpoints válidos")

        return last_valid

    def clear(self) -> None:
        """Borra el archivo de estado."""
        if self._path.exists():
            self._path.unlink()
            log.info("estado borrado", path=str(self._path))


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: restaurar Wallet desde Checkpoint
# ══════════════════════════════════════════════════════════════════════════════

def restore_wallet_from_checkpoint(
    checkpoint:     Checkpoint,
    max_posiciones: int,
) -> MemoryWallet:
    """
    Reconstruye una MemoryWallet desde un Checkpoint guardado.
    Llamar en el arranque del live_trader tras restore().

    Retorna una MemoryWallet con el estado completo restaurado.
    Los runners pueden hacer cast a JSONWallet si necesitan persistencia.
    """
    from actors.wallet import Position
    from collections   import deque

    wallet = MemoryWallet(
        usdt_inicial   = checkpoint.usdt_balance,
        max_posiciones = max_posiciones,
    )
    # Sobreescribir estado interno directamente
    wallet._usdt          = checkpoint.usdt_balance
    wallet._btc_libre     = checkpoint.btc_libre
    wallet._slot_usdt     = checkpoint.slot_usdt
    wallet._btc_por_venta = checkpoint.btc_por_venta
    wallet._posiciones    = deque(
        Position(
            entry_price = p["entry_price"],
            btc         = p["btc"],
            opened_at   = p["opened_at"],
        )
        for p in checkpoint.positions
    )
    log.info(
        "Wallet restaurada desde checkpoint",
        usdt=f"{wallet.get_usdt_balance():.2f}",
        positions=wallet.positions_count,
        slot=f"{wallet.get_slot_usdt():.2f}",
    )
    return wallet


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_state_manager() -> StateManager:
    """
    Construye el StateManager según mode_config.

    Modos (mode_config.py):
        USE_LIVE_CLOCK = False  →  MemoryStateManager (backtest, sin I/O)
        USE_LIVE_CLOCK = True   →  JSONStateManager   (producción)
    """
    try:
        import mode_config as MC
        use_live = getattr(MC, "USE_LIVE_CLOCK", False)
    except ImportError:
        use_live = False

    if not use_live:
        log.info("StateManager modo MEMORY (backtest)")
        return MemoryStateManager()

    try:
        import config_local as CL
        state_path = getattr(CL, "STATE_PATH", "state/trading_state.jsonl")
    except ImportError:
        state_path = "state/trading_state.jsonl"

    log.info("StateManager modo JSON", path=state_path)
    return JSONStateManager(state_path=state_path)
