"""
clock.py — Actor 4: Reloj / Director de ciclos
═══════════════════════════════════════════════
Responsabilidad única: decidir CUÁNDO se ejecuta cada ciclo de la estrategia.

Interfaz abstracta Clock
─────────────────────────
  tick()         →  Candle | None   (None = fin del stream)
  reset()        →  None            (vuelve al inicio — usado en grid search)
  is_live        →  bool            (False en backtest, True en producción)

Implementaciones
─────────────────
  LocalClock   →  itera velas desde un PriceFeed local (backtest)
  LiveClock    →  stream en tiempo real via BinanceWSFeed (actors/live_clock.py)

Factory
────────
  build_clock(feed, mode, ...)  →  Clock
    mode="local"  →  LocalClock  (default, backtest)
    mode="live"   →  LiveClock   (producción/testnet)

  No lee mode_config — el runner decide el modo explícitamente.

Uso en runners
───────────────
  # Backtest
  clock = LocalClock(feed, start, end)
  while (candle := clock.tick()) is not None:
      signal = strategy.on_candle(candle)
      ...

  # Producción (mismo código, distinto clock)
  from actors.live_clock import LiveClock
  clock = LiveClock(feed, symbol)
  while (candle := clock.tick()) is not None:
      signal = strategy.on_candle(candle)
      ...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, Optional

from actors.price_feed  import Candle, PriceFeed
from support.logger     import get_logger
from support.time_utils import TimeInput, to_epoch_s, to_iso

log = get_logger("clock")


# ══════════════════════════════════════════════════════════════════════════════
# INTERFAZ ABSTRACTA
# ══════════════════════════════════════════════════════════════════════════════

class Clock(ABC):
    """
    Contrato que deben cumplir todas las implementaciones del reloj.
    La estrategia y los runners solo conocen esta interfaz.
    """

    @abstractmethod
    def tick(self) -> Optional[Candle]:
        """
        Retorna la siguiente vela disponible.
        Retorna None cuando el stream terminó (fin del backtest
        o señal de parada en producción).
        """

    @abstractmethod
    def reset(self) -> None:
        """
        Reinicia el clock al estado inicial.
        Usado en grid search para reutilizar el mismo clock entre runs.
        En producción no tiene efecto.
        """

    @property
    @abstractmethod
    def is_live(self) -> bool:
        """True si el clock está conectado a datos en tiempo real."""

    def __iter__(self) -> Iterator[Candle]:
        """Permite usar el clock como iterador: for candle in clock."""
        while (candle := self.tick()) is not None:
            yield candle


# ══════════════════════════════════════════════════════════════════════════════
# IMPLEMENTACIÓN LOCAL: LocalClock
# ══════════════════════════════════════════════════════════════════════════════

class LocalClock(Clock):
    """
    Itera velas desde un PriceFeed local (SQLite o CSV).
    Cada llamada a tick() retorna la siguiente vela en orden cronológico.

    Diseñado para:
      · Backtest completo (runner itera vela a vela)
      · Grid search (reset() entre runs sin recargar datos)
      · Tests unitarios (inyección de velas sintéticas)
    """

    def __init__(
        self,
        feed:   PriceFeed,
        start:  TimeInput,
        end:    TimeInput,
        symbol: str = "BTCUSDT",
    ) -> None:
        self._feed    = feed
        self._start   = start
        self._end     = end
        self._symbol  = symbol
        self._candles: list[Candle] = []
        self._cursor:  int  = 0
        self._loaded:  bool = False

        log.info(
            "LocalClock inicializado",
            start=to_iso(to_epoch_s(start)),
            end=to_iso(to_epoch_s(end)),
            symbol=symbol,
        )

    # ── Interfaz ──────────────────────────────────────────────────────────────

    @property
    def is_live(self) -> bool:
        return False

    def tick(self) -> Optional[Candle]:
        """
        Retorna la siguiente vela. Carga el dataset en el primer tick()
        (lazy loading — no carga hasta que se necesita).
        """
        if not self._loaded:
            self._load()
        if self._cursor >= len(self._candles):
            return None
        candle = self._candles[self._cursor]
        self._cursor += 1
        return candle

    def reset(self) -> None:
        """Vuelve al inicio sin recargar datos del disco."""
        self._cursor = 0
        log.debug("LocalClock reseteado", symbol=self._symbol)

    # ── Propiedades informativas ───────────────────────────────────────────────

    @property
    def total_candles(self) -> int:
        """Número total de velas en el rango (disponible tras primer tick)."""
        return len(self._candles)

    @property
    def candles_remaining(self) -> int:
        """Velas que quedan por procesar."""
        return max(0, len(self._candles) - self._cursor)

    @property
    def progress_pct(self) -> float:
        """Porcentaje de avance del backtest [0.0 - 100.0]."""
        if not self._candles:
            return 0.0
        return self._cursor / len(self._candles) * 100.0

    def peek(self) -> Optional[Candle]:
        """Mira la próxima vela sin consumirla."""
        if not self._loaded:
            self._load()
        if self._cursor >= len(self._candles):
            return None
        return self._candles[self._cursor]

    # ── Helper privado ────────────────────────────────────────────────────────

    def _load(self) -> None:
        self._candles = self._feed.get_candles(
            start=self._start,
            end=self._end,
            symbol=self._symbol,
        )
        self._cursor = 0
        self._loaded = True
        log.info(
            "velas cargadas en LocalClock",
            total=len(self._candles),
            symbol=self._symbol,
        )


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def build_clock(
    feed:   PriceFeed,
    mode:   str       = "local",
    start:  TimeInput = None,
    end:    TimeInput = None,
    symbol: str       = None,
) -> Clock:
    """
    Helper para construir un Clock sin instanciar manualmente.
    No lee mode_config — el runner decide el modo explícitamente.

    mode="local"  →  LocalClock  (default, backtest)
    mode="live"   →  LiveClock   (producción/testnet)

    start y end son requeridos para mode="local".
    Si no se pasan, se leen desde config_local como fallback.

    Para control fino, instanciar directamente:
        clock = LocalClock(feed, start=CL.FECHA_INICIO, end=CL.FECHA_FIN)
        clock = LiveClock(feed, symbol="BTCUSDT")
    """
    try:
        import config_local as CL
        _start  = start  or getattr(CL, "FECHA_INICIO", None)
        _end    = end    or getattr(CL, "FECHA_FIN",    None)
        _symbol = symbol or getattr(CL, "SYMBOL",       "BTCUSDT")
    except ImportError:
        _start  = start
        _end    = end
        _symbol = symbol or "BTCUSDT"

    if mode == "live":
        from actors.live_clock import LiveClock
        log.info("Clock → LiveClock", symbol=_symbol)
        return LiveClock(feed=feed, symbol=_symbol)

    # default: local
    if not _start or not _end:
        raise ValueError(
            "start y end son requeridos para LocalClock. "
            "Pasalos explícitamente o definí FECHA_INICIO/FECHA_FIN en config_local.py."
        )

    log.info("Clock → LocalClock", start=_start, end=_end)
    return LocalClock(feed=feed, start=_start, end=_end, symbol=_symbol)