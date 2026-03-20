"""
base_strategy.py — Interfaz abstracta de estrategia
═════════════════════════════════════════════════════
Contrato que deben cumplir todas las estrategias del sistema.

La estrategia recibe velas y decide qué señal emitir.
No sabe nada del modo de ejecución (backtest vs live),
ni de qué implementación de actor está en uso.
Solo conoce las interfaces abstractas de los actores.

Ciclo de vida
──────────────
  1. on_start()            → inicialización (cargar modelos, warm-up, etc.)
  2. on_candle(candle)     → señal por cada vela recibida del Clock
  3. on_stop()             → limpieza al finalizar

Señal
──────
  Signal es un dataclass simple:
    side   = BUY | SELL | HOLD
    price  = precio de ejecución sugerido
    reason = motivo legible (para logs y JSON)

  La estrategia emite la señal.
  El runner decide si ejecutarla (consultando RiskManager y OrderBook).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from actors.price_feed import Candle
from actors.wallet     import Wallet
from actors.order_book import OrderBook, OrderSide
from support.logger    import get_logger

log = get_logger("strategy")


# ══════════════════════════════════════════════════════════════════════════════
# TIPOS
# ══════════════════════════════════════════════════════════════════════════════

class SignalSide(str, Enum):
    BUY  = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Signal:
    """
    Señal emitida por la estrategia tras procesar una vela.

    side:   qué hacer
    price:  precio de ejecución sugerido (close de la vela por defecto)
    reason: descripción legible — aparece en logs y en el JSON de resultados
    score:  valor numérico opcional de la señal (ej: score 0-100 del compuesto)
    """
    side:   SignalSide
    price:  float
    reason: str               = ""
    score:  Optional[float]   = None

    @property
    def is_actionable(self) -> bool:
        """True si la señal requiere una operación (no HOLD)."""
        return self.side != SignalSide.HOLD

    def to_order_side(self) -> Optional[OrderSide]:
        """Convierte a OrderSide para pasarle al OrderBook."""
        if self.side == SignalSide.BUY:
            return OrderSide.BUY
        if self.side == SignalSide.SELL:
            return OrderSide.SELL
        return None


HOLD = Signal(side=SignalSide.HOLD, price=0.0, reason="sin_señal")


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ ABSTRACTA
# ══════════════════════════════════════════════════════════════════════════════

class BaseStrategy(ABC):
    """
    Contrato base para todas las estrategias.
    Las subclases implementan on_candle() con su lógica propia.
    Los parámetros de la estrategia viven dentro de la subclase.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self._candles_seen: int = 0
        log.info("estrategia inicializada", nombre=name)

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def on_start(self, wallet: Wallet) -> None:
        """
        Llamar una vez antes del primer tick().
        Usar para warm-up de indicadores, carga de modelos, etc.
        Implementación por defecto: no hace nada.
        """

    @abstractmethod
    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Procesa una vela y retorna la señal correspondiente.
        Llamado por el runner en cada tick() del Clock.

        No ejecutar órdenes aquí — solo emitir señales.
        El runner se encarga de pasar la señal al OrderBook.
        """

    def on_stop(self, wallet: Wallet) -> None:
        """
        Llamar una vez al finalizar.
        Usar para guardar estado, cerrar conexiones, etc.
        Implementación por defecto: no hace nada.
        """

    # ── Helpers disponibles para las subclases ────────────────────────────────

    @property
    def candles_seen(self) -> int:
        """Número de velas procesadas desde on_start()."""
        return self._candles_seen

    def _tick(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Wrapper interno llamado por el runner.
        Incrementa el contador y delega a on_candle().
        No sobreescribir en las subclases.
        """
        self._candles_seen += 1
        return self.on_candle(candle, wallet)
