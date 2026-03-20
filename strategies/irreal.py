"""
irreal.py — Estrategia Irreal (Oráculo Perfecto)
══════════════════════════════════════════════════
Benchmark teórico: compra en cada mínimo local, vende en cada máximo local.
Requiere ver el futuro → imposible en producción → útil solo como techo.

Adaptación al nuevo sistema de actores
────────────────────────────────────────
La lógica de detección es idéntica al Backtest_irreal.py original.
Lo que cambia es la arquitectura:
  · La estrategia solo emite Signal (BUY / SELL / HOLD)
  · El runner pasa la señal al OrderBook
  · El OrderBook ejecuta y notifica a la Wallet
  · La Wallet mantiene la lógica de slots

Parámetros propios de la estrategia
──────────────────────────────────────
Viven dentro de esta clase — no en config_local.py.
Se pueden cambiar al instanciar: IrrealStrategy(ventana=15)

  ventana:       int   = 10   velas a cada lado para confirmar extremo
  precio_compra: str   = "low"   precio de ejecución en BUY  (low/close/open)
  precio_venta:  str   = "high"  precio de ejecución en SELL (high/close/open)

Nota sobre el oráculo
──────────────────────
La detección multi-escala requiere VER velas futuras para confirmar que
una vela es un mínimo/máximo local. Por eso:
  · Las primeras `ventana` velas dan HOLD (no hay contexto pasado)
  · Las últimas `ventana` velas dan HOLD (no hay contexto futuro)
  · Solo las velas intermedias pueden generar BUY/SELL

Esto es intencional: es exactamente la definición de oráculo perfecto.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional

from actors.price_feed   import Candle
from actors.wallet       import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger      import get_logger

log = get_logger("irreal")


class IrrealStrategy(BaseStrategy):
    """
    Detecta mínimos y máximos locales con ventana de N velas a cada lado.
    Emite BUY en mínimos locales y SELL en máximos locales.

    Usa un buffer deslizante de (2*ventana + 1) velas para detectar
    el extremo central sin mirar más allá de lo necesario.
    """

    # ── Configuración propia de la estrategia ─────────────────────────────────
    DEFAULT_VENTANA       = 10
    DEFAULT_PRECIO_COMPRA = "low"    # "low" | "close" | "open"
    DEFAULT_PRECIO_VENTA  = "high"   # "high" | "close" | "open"

    def __init__(
        self,
        ventana:       int = DEFAULT_VENTANA,
        precio_compra: str = DEFAULT_PRECIO_COMPRA,
        precio_venta:  str = DEFAULT_PRECIO_VENTA,
    ) -> None:
        super().__init__(name="Irreal-OráculoPerfecto")

        # Validaciones
        if ventana < 1:
            raise ValueError(f"ventana debe ser >= 1, got {ventana}")
        if precio_compra not in ("low", "close", "open"):
            raise ValueError(f"precio_compra inválido: {precio_compra}")
        if precio_venta not in ("high", "close", "open"):
            raise ValueError(f"precio_venta inválido: {precio_venta}")

        self.ventana       = ventana
        self.precio_compra = precio_compra
        self.precio_venta  = precio_venta

        # Buffer circular de tamaño (2*ventana + 1)
        # Cuando está lleno, la vela central es candidata a extremo local
        self._buffer: Deque[Candle] = deque(maxlen=2 * ventana + 1)

        # Cola de señales pendientes de entregar
        # (se generan cuando el buffer se llena y la vela candidata es extremo)
        self._pending: Deque[Signal] = deque()

        log.info(
            "IrrealStrategy configurada",
            ventana=ventana,
            precio_compra=precio_compra,
            precio_venta=precio_venta,
        )

    # ── Interfaz BaseStrategy ─────────────────────────────────────────────────

    def on_start(self, wallet: Wallet) -> None:
        self._buffer.clear()
        self._pending.clear()
        log.info("IrrealStrategy iniciada")

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Agrega la vela al buffer.
        Cuando el buffer se llena, evalúa si la vela central es extremo.
        Si hay señales pendientes de ciclos anteriores, las entrega primero.
        """
        # Entregar señales pendientes de ciclos anteriores
        if self._pending:
            return self._pending.popleft()

        self._buffer.append(candle)

        # Necesitamos buffer lleno para evaluar la vela central
        if len(self._buffer) < self._buffer.maxlen:
            return HOLD

        signal = self._evaluar_central()
        return signal if signal is not None else HOLD

    def on_stop(self, wallet: Wallet) -> None:
        """
        Las últimas `ventana` velas no pudieron ser evaluadas
        (no tienen contexto futuro completo) — esto es correcto.
        """
        n_sin_evaluar = min(len(self._buffer), self.ventana)
        if n_sin_evaluar > 0:
            log.info(
                "IrrealStrategy detenida",
                velas_sin_evaluar=n_sin_evaluar,
                nota="últimas N velas sin contexto futuro — comportamiento esperado",
            )

    # ── Lógica de detección ───────────────────────────────────────────────────

    def _evaluar_central(self) -> Optional[Signal]:
        """
        Evalúa si la vela central del buffer es un mínimo o máximo local.
        La vela central es buffer[ventana] (índice del medio).
        """
        buf    = list(self._buffer)
        centro = self.ventana          # índice de la vela candidata
        vela_c = buf[centro]

        low_c  = vela_c.low
        high_c = vela_c.high

        vecinos = [buf[j] for j in range(len(buf)) if j != centro]

        es_minimo = all(low_c  <= v.low  for v in vecinos)
        es_maximo = all(high_c >= v.high for v in vecinos)

        # Si es ambas simultáneamente (vela plana extrema): priorizar BUY
        if es_minimo:
            precio_exec = self._precio_candle(vela_c, self.precio_compra)
            log.debug(
                "mínimo local detectado",
                ts=vela_c.iso(),
                low=low_c,
                precio_exec=precio_exec,
            )
            return Signal(
                side   = SignalSide.BUY,
                price  = precio_exec,
                reason = f"minimo_local(ventana={self.ventana})",
                score  = 100.0,
            )

        if es_maximo:
            precio_exec = self._precio_candle(vela_c, self.precio_venta)
            log.debug(
                "máximo local detectado",
                ts=vela_c.iso(),
                high=high_c,
                precio_exec=precio_exec,
            )
            return Signal(
                side   = SignalSide.SELL,
                price  = precio_exec,
                reason = f"maximo_local(ventana={self.ventana})",
                score  = 100.0,
            )

        return None

    @staticmethod
    def _precio_candle(candle: Candle, campo: str) -> float:
        """Retorna el precio de ejecución según el campo configurado."""
        return {
            "low":   candle.low,
            "high":  candle.high,
            "open":  candle.open,
            "close": candle.close,
        }[campo]

    # ── Estadísticas ──────────────────────────────────────────────────────────

    def describe(self) -> dict:
        """Retorna la configuración de la estrategia como dict (para el JSON)."""
        return {
            "estrategia":   self.name,
            "ventana":      self.ventana,
            "precio_compra": self.precio_compra,
            "precio_venta":  self.precio_venta,
        }
