"""
strategies/grid_extremes.py — GridExtremes Strategy
═════════════════════════════════════════════════════
Usa el oráculo de extremos locales de IrrealStrategy como referencias
de precio para definir niveles de entrada y salida en grilla incremental.

Lógica de señales
──────────────────
  BUY  levels : candle.low  <= last_top_high   * (1 - i * drop_pct_buy  / 100)
  SELL levels : candle.high >= last_bottom_low  * (1 + i * rise_pct_sell / 100)

  i ∈ [1, MAX_POSICIONES]   (leído desde config_local)

El precio de ejecución de cada señal es exactamente el precio del nivel
cruzado, no el close de la vela. Esto es consistente con IrrealStrategy
que usa low/high del extremo como precio de ejecución.

Confirmación de extremos (oráculo)
────────────────────────────────────
  Mismo buffer circular de IrrealStrategy: ventana velas a cada lado.
  El delay es inherente — la vela central no se puede confirmar hasta
  tener ventana velas de contexto futuro.

  Si es_bottom Y es_top simultáneamente (vela plana extrema):
  se prioriza bottom, consistente con IrrealStrategy.

Reset de niveles
─────────────────
  Nuevo top  confirmado → triggered_buy.clear()  + recalcular referencia BUY
  Nuevo bottom confirmado → triggered_sell.clear() + recalcular referencia SELL
  Las posiciones abiertas NO se cierran — la Wallet las gestiona normalmente.

RETROACTIVE=False (default)
────────────────────────────
  Al confirmar un extremo, se inspeccionan las velas posteriores al
  mismo que ya pasaron (buffer[ventana+1:-1]) y se marcan como
  disparados los niveles que ya fueron cruzados durante el delay,
  sin emitir señal. Evita órdenes fantasma retroactivas.

Prioridad BUY/SELL simultáneos
────────────────────────────────
  positions == MAX_POSICIONES → solo SELL (no se puede comprar más)
  positions == 0              → solo BUY  (no hay posiciones para vender)
  intermedio + close >= open  → SELL primero, luego BUY (subió antes)
  intermedio + close <  open  → BUY  primero, luego SELL (bajó antes)

Cola de señales pendientes
───────────────────────────
  Si una vela cruza N niveles, se entrega el primero inmediatamente
  y los N-1 restantes se encolan en _pending. Los ticks siguientes
  entregan las señales pendientes ANTES de procesar la nueva vela.
  Mismo patrón que IrrealStrategy._pending.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, List, Optional, Set

from actors.price_feed        import Candle
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("grid_extremes")


class GridExtremesStrategy(BaseStrategy):
    """
    Grilla de compra/venta basada en extremos locales confirmados por oráculo.

    Parámetros
    ──────────
    ventana        : velas a cada lado para confirmar extremo (default 10, igual que irreal)
    drop_pct_buy   : % de caída por nivel desde el último top   (ej: 5.0 → niveles en 5, 10, 15...)
    rise_pct_sell  : % de subida por nivel desde el último bottom (ej: 5.0 → niveles en 5, 10, 15...)
    retroactive    : False (default) = no disparar niveles cruzados durante el delay del oráculo
    """

    DEFAULT_VENTANA      = 10
    DEFAULT_DROP_PCT     = 5.0
    DEFAULT_RISE_PCT     = 5.0
    DEFAULT_RETROACTIVE  = False

    def __init__(
        self,
        ventana:       int   = DEFAULT_VENTANA,
        drop_pct_buy:  float = DEFAULT_DROP_PCT,
        rise_pct_sell: float = DEFAULT_RISE_PCT,
        retroactive:   bool  = DEFAULT_RETROACTIVE,
    ) -> None:
        super().__init__(name="GridExtremes")

        if ventana < 1:
            raise ValueError(f"ventana >= 1, got {ventana}")
        if not (0 < drop_pct_buy < 100):
            raise ValueError(f"drop_pct_buy en (0, 100), got {drop_pct_buy}")
        if not (0 < rise_pct_sell < 100):
            raise ValueError(f"rise_pct_sell en (0, 100), got {rise_pct_sell}")

        self.ventana       = ventana
        self.drop_pct_buy  = drop_pct_buy
        self.rise_pct_sell = rise_pct_sell
        self.retroactive   = retroactive

        # MAX_POSICIONES define cuántos niveles puede haber por ciclo.
        # Se lee desde config_local para mantener consistencia con el resto
        # del sistema (Wallet, OrderBook, RiskManager usan el mismo valor).
        try:
            import config_local as CL
            self._max_levels: int = int(getattr(CL, "MAX_POSICIONES", 5))
        except ImportError:
            self._max_levels = 5

        # ── Buffer circular del oráculo ──────────────────────────────────────
        # Idéntico al de IrrealStrategy: la vela central (índice ventana)
        # se evalúa como candidata a extremo local cuando el buffer está lleno.
        self._buffer: Deque[Candle] = deque(maxlen=2 * ventana + 1)

        # ── Referencias de extremos confirmados ──────────────────────────────
        self._last_top_high:   Optional[float] = None   # high del último top local
        self._last_bottom_low: Optional[float] = None   # low del último bottom local

        # ── Sets de niveles ya disparados en el ciclo actual ─────────────────
        # Se vacían cuando se confirma un nuevo extremo del mismo tipo.
        self._triggered_buy:  Set[int] = set()
        self._triggered_sell: Set[int] = set()

        # ── Cola de señales pendientes de entregar ───────────────────────────
        # Cuando una vela cruza múltiples niveles, se entrega la primera señal
        # y las demás se encolan aquí para entregarlas en ticks sucesivos.
        self._pending: Deque[Signal] = deque()

        log.info(
            "GridExtremesStrategy configurada",
            ventana        = ventana,
            drop_pct_buy   = f"{drop_pct_buy}%",
            rise_pct_sell  = f"{rise_pct_sell}%",
            retroactive    = retroactive,
            max_levels     = self._max_levels,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # INTERFAZ BaseStrategy
    # ══════════════════════════════════════════════════════════════════════════

    def on_start(self, wallet: Wallet, **kwargs) -> None:
        self._buffer.clear()
        self._pending.clear()
        self._last_top_high   = None
        self._last_bottom_low = None
        self._triggered_buy.clear()
        self._triggered_sell.clear()
        log.info(
            "GridExtremesStrategy iniciada",
            max_levels    = self._max_levels,
            drop_pct_buy  = f"{self.drop_pct_buy}%",
            rise_pct_sell = f"{self.rise_pct_sell}%",
        )

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Ciclo por vela:
          1. Si hay señales pendientes de velas anteriores, entregarlas primero.
          2. Agregar vela al buffer del oráculo.
          3. Si el buffer está lleno, evaluar la vela central como extremo.
          4. Verificar qué niveles cruza la vela actual.
          5. Entregar primera señal; encolar el resto en _pending.
        """
        # ── 1. Señales pendientes ─────────────────────────────────────────────
        if self._pending:
            return self._pending.popleft()

        # ── 2 + 3. Buffer y oráculo ───────────────────────────────────────────
        self._buffer.append(candle)
        if len(self._buffer) == self._buffer.maxlen:
            self._evaluate_central()

        # ── 4. Verificar niveles ──────────────────────────────────────────────
        new_signals = self._check_levels(candle, wallet)
        if not new_signals:
            return HOLD

        # ── 5. Entregar + encolar ─────────────────────────────────────────────
        first = new_signals[0]
        self._pending.extend(new_signals[1:])
        return first

    def on_stop(self, wallet: Wallet) -> None:
        n_sin_evaluar = min(len(self._buffer), self.ventana)
        log.info(
            "GridExtremesStrategy detenida",
            velas_procesadas  = self.candles_seen,
            velas_sin_evaluar = n_sin_evaluar,
            ultimo_top_high   = self._last_top_high,
            ultimo_bottom_low = self._last_bottom_low,
            nota = "últimas N velas sin contexto futuro — comportamiento esperado",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # ORÁCULO: DETECCIÓN DE EXTREMOS
    # ══════════════════════════════════════════════════════════════════════════

    def _evaluate_central(self) -> None:
        """
        Evalúa si la vela central del buffer es un mínimo o máximo local.
        Lógica idéntica a IrrealStrategy._evaluar_central().

        La vela central es buffer[ventana] (índice del medio del buffer lleno).
        Se confirma como extremo si su low/high es ≤/≥ al de TODAS las demás
        velas del buffer (condición ≤/≥ incluye empates, como en irreal).

        Prioridad ante ambas condiciones simultáneas: bottom > top (igual que irreal).
        """
        buf    = list(self._buffer)
        centro = self.ventana
        vela_c = buf[centro]
        vecinos = [buf[j] for j in range(len(buf)) if j != centro]

        es_bottom = all(vela_c.low  <= v.low  for v in vecinos)
        es_top    = all(vela_c.high >= v.high for v in vecinos)

        if es_bottom:
            self._confirm_bottom(vela_c)
        elif es_top:
            self._confirm_top(vela_c)

    def _confirm_bottom(self, confirmed_candle: Candle) -> None:
        """
        Nuevo mínimo local confirmado.

        Acciones:
          · Actualiza _last_bottom_low con el low de la vela confirmada.
          · Vacía _triggered_sell (ciclo nuevo para señales SELL).
          · Si RETROACTIVE=False, inspecciona las velas que ya pasaron
            durante el delay del oráculo (buffer[ventana+1:-1]) y marca
            como disparados los niveles SELL ya cruzados, sin emitir señal.
            El slice :-1 excluye la vela actual, que sí puede disparar.
        """
        self._last_bottom_low = confirmed_candle.low
        self._triggered_sell.clear()

        if not self.retroactive:
            for sub in list(self._buffer)[self.ventana + 1 : -1]:
                for i in range(1, self._max_levels + 1):
                    if i not in self._triggered_sell:
                        lvl = self._last_bottom_low * (1.0 + i * self.rise_pct_sell / 100.0)
                        if sub.high >= lvl:
                            self._triggered_sell.add(i)
                            log.debug(
                                "nivel SELL retroactivo marcado (sin señal)",
                                nivel       = i,
                                precio_lvl  = f"{lvl:.2f}",
                                vela_ts     = sub.iso(),
                                bottom_ref  = f"{self._last_bottom_low:.2f}",
                            )

        log.debug(
            "bottom local confirmado → referencia SELL actualizada",
            ts            = confirmed_candle.iso(),
            low           = confirmed_candle.low,
            triggered_sell_reseteados = True,
        )

    def _confirm_top(self, confirmed_candle: Candle) -> None:
        """
        Nuevo máximo local confirmado.

        Acciones:
          · Actualiza _last_top_high con el high de la vela confirmada.
          · Vacía _triggered_buy (ciclo nuevo para señales BUY).
          · Si RETROACTIVE=False, marca como disparados los niveles BUY
            ya cruzados durante el delay (buffer[ventana+1:-1]).
        """
        self._last_top_high = confirmed_candle.high
        self._triggered_buy.clear()

        if not self.retroactive:
            for sub in list(self._buffer)[self.ventana + 1 : -1]:
                for i in range(1, self._max_levels + 1):
                    if i not in self._triggered_buy:
                        lvl = self._last_top_high * (1.0 - i * self.drop_pct_buy / 100.0)
                        if sub.low <= lvl:
                            self._triggered_buy.add(i)
                            log.debug(
                                "nivel BUY retroactivo marcado (sin señal)",
                                nivel      = i,
                                precio_lvl = f"{lvl:.2f}",
                                vela_ts    = sub.iso(),
                                top_ref    = f"{self._last_top_high:.2f}",
                            )

        log.debug(
            "top local confirmado → referencia BUY actualizada",
            ts           = confirmed_candle.iso(),
            high         = confirmed_candle.high,
            triggered_buy_reseteados = True,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # VERIFICACIÓN DE NIVELES
    # ══════════════════════════════════════════════════════════════════════════

    def _check_levels(self, candle: Candle, wallet: Wallet) -> List[Signal]:
        """
        Comprueba qué niveles BUY y SELL cruza la vela actual.
        Retorna la lista ordenada de señales a emitir según la lógica
        de prioridad de posiciones y dirección de vela.

        El precio de ejecución de cada señal es el precio exacto del nivel,
        no el close de la vela (consistente con irreal que usa low/high exactos).
        """
        buy_signals:  List[Signal] = []
        sell_signals: List[Signal] = []

        # ── Niveles BUY (requiere referencia de top confirmada) ───────────────
        if self._last_top_high is not None and wallet.positions_count < self._max_levels:
            for i in range(1, self._max_levels + 1):
                if i in self._triggered_buy:
                    continue
                lvl = self._last_top_high * (1.0 - i * self.drop_pct_buy / 100.0)
                if candle.low <= lvl:
                    self._triggered_buy.add(i)
                    buy_signals.append(Signal(
                        side   = SignalSide.BUY,
                        price  = round(lvl, 8),
                        reason = (
                            f"grid_buy_nivel_{i}"
                            f"(top={self._last_top_high:.2f}"
                            f",drop={i * self.drop_pct_buy:.1f}%)"
                        ),
                        score  = float(i),
                    ))
                    log.debug(
                        "nivel BUY disparado",
                        nivel     = i,
                        precio    = f"{lvl:.2f}",
                        candle_ts = candle.iso(),
                        top_ref   = f"{self._last_top_high:.2f}",
                    )

        # ── Niveles SELL (requiere referencia de bottom confirmada) ───────────
        if self._last_bottom_low is not None and wallet.positions_count > 0:
            for i in range(1, self._max_levels + 1):
                if i in self._triggered_sell:
                    continue
                lvl = self._last_bottom_low * (1.0 + i * self.rise_pct_sell / 100.0)
                if candle.high >= lvl:
                    self._triggered_sell.add(i)
                    sell_signals.append(Signal(
                        side   = SignalSide.SELL,
                        price  = round(lvl, 8),
                        reason = (
                            f"grid_sell_nivel_{i}"
                            f"(bottom={self._last_bottom_low:.2f}"
                            f",rise={i * self.rise_pct_sell:.1f}%)"
                        ),
                        score  = float(i),
                    ))
                    log.debug(
                        "nivel SELL disparado",
                        nivel      = i,
                        precio     = f"{lvl:.2f}",
                        candle_ts  = candle.iso(),
                        bottom_ref = f"{self._last_bottom_low:.2f}",
                    )

        # ── Sin señales ───────────────────────────────────────────────────────
        if not buy_signals and not sell_signals:
            return []
        if not buy_signals:
            return sell_signals
        if not sell_signals:
            return buy_signals

        # ── Señales de ambos tipos: aplicar lógica de prioridad ───────────────
        positions = wallet.positions_count

        if positions >= self._max_levels:
            # Posiciones llenas: en la vida real no se podría comprar → solo SELL
            log.debug(
                "prioridad: solo SELL (posiciones == max_levels)",
                positions  = positions,
                max_levels = self._max_levels,
            )
            return sell_signals

        elif positions == 0:
            # Sin posiciones: no hay nada que vender → solo BUY
            log.debug("prioridad: solo BUY (positions == 0)")
            return buy_signals

        else:
            # Posiciones intermedias: orden según dirección de la vela
            if candle.close >= candle.open:
                # Vela alcista o doji: se asume que el precio subió primero
                # (tocó el high antes que el low) → SELL primero, luego BUY
                log.debug(
                    "prioridad: SELL → BUY (vela alcista)",
                    open  = candle.open,
                    close = candle.close,
                )
                return sell_signals + buy_signals
            else:
                # Vela bajista: el precio bajó primero
                # (tocó el low antes que el high) → BUY primero, luego SELL
                log.debug(
                    "prioridad: BUY → SELL (vela bajista)",
                    open  = candle.open,
                    close = candle.close,
                )
                return buy_signals + sell_signals

    # ══════════════════════════════════════════════════════════════════════════
    # DESCRIBE
    # ══════════════════════════════════════════════════════════════════════════

    def describe(self) -> dict:
        return {
            "estrategia":        self.name,
            "ventana":           self.ventana,
            "drop_pct_buy":      self.drop_pct_buy,
            "rise_pct_sell":     self.rise_pct_sell,
            "retroactive":       self.retroactive,
            "max_levels":        self._max_levels,
            "ultimo_top_high":   self._last_top_high,
            "ultimo_bottom_low": self._last_bottom_low,
            # Campos requeridos por el esquema del Graficador / JSON summary
            "rsi_length":        "N/A",
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "N":                 self.ventana,
            "guardia_compra":    True,
            "guardia_venta":     True,
        }
