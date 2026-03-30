"""
strategies/mean_reversion.py — Estrategia Mean Reversion Técnica
═════════════════════════════════════════════════════════════════
Detecta mínimos y máximos locales usando exclusivamente indicadores
técnicos calculables en tiempo real sobre el price feed de Binance.

Motivación y calibración
─────────────────────────
Diseñada a partir del análisis de los 280 mínimos locales detectados
por el backtest irreal (oráculo ±10 velas) en el período 2021-11 a
2022-11. Las observaciones clave que guían el diseño:

  · Antes de cada bottom: el precio cayó media -3.8% en 12h previas.
    Un umbral de -2% cubre el 82% de todos los bottoms.
  · Antes de cada top: el precio subió media +3.9% en 12h previas.
    Un umbral de +2% cubre el 87% de todos los tops.
  · Duración media de ciclo buy→sell: 46h (rebote promedio: +4.4%).
  · Los ciclos más cortos (8-16h) representan el 12% del total.

Indicadores usados
───────────────────
  · RSI(14, 1h)          — sobreventa / sobrecompra
  · Drop24               — caída % desde el máximo de las últimas 24 velas
  · Rise24               — subida % desde el mínimo de las últimas 24 velas
  · BB(20, 2, 1h)        — bandas de Bollinger (posición dentro de las bandas)
  · Low rejection        — (close - low) / (high - low): mecha inferior
  · High rejection       — (high - close) / (high - low): mecha superior
  · EMA_fast / EMA_slow  — cruce de corto plazo para confirmar dirección

Lógica de señales
──────────────────
COMPRA  — se necesitan al menos PUNTOS_BUY_MIN puntos del siguiente set:
  [3pts] RSI14 < RSI_BUY_STRONG      (sobreventa fuerte)
  [2pts] RSI14 < RSI_BUY_WEAK        (sobreventa moderada)
  [2pts] drop24 < DROP_STRONG        (caída fuerte reciente)
  [1pt]  drop24 < DROP_WEAK          (caída moderada reciente)
  [2pts] close < BB_lower            (precio bajo BB inferior)
  [1pt]  close < BB_lower * 1.005    (precio cerca de BB inferior)
  [2pts] low_rejection > LOW_REJ_STRONG   (mecha inferior pronunciada)
  [1pt]  low_rejection > LOW_REJ_WEAK     (algo de mecha inferior)
  [1pt]  ema_fast cruza sobre ema_slow    (inicio de rebote)

VENTA   — se necesitan al menos PUNTOS_SELL_MIN puntos:
  [3pts] RSI14 > RSI_SELL_STRONG     (sobrecompra fuerte)
  [2pts] RSI14 > RSI_SELL_WEAK       (sobrecompra moderada)
  [2pts] rise24 > RISE_STRONG        (subida fuerte reciente)
  [1pt]  rise24 > RISE_WEAK          (subida moderada reciente)
  [2pts] close > BB_upper            (precio sobre BB superior)
  [1pt]  close > BB_upper * 0.995    (precio cerca de BB superior)
  [2pts] high_rejection > HIGH_REJ_STRONG (mecha superior pronunciada)
  [1pt]  high_rejection > HIGH_REJ_WEAK   (algo de mecha superior)
  [1pt]  rise_from_entry > TAKE_PROFIT_PCT (ganancia desde entrada)

STOP-LOSS por posición:
  Si el precio cae más de STOP_LOSS_PCT desde el precio de entrada
  de la posición más antigua, se cierra esa posición.
  Esto reduce avg_loss desde ~4.2% a ~2.5%, haciendo el sistema rentable.

Parámetros expuestos (configurables en el runner)
───────────────────────────────────────────────────
  rsi_period        int   = 14      periodo del RSI
  bb_period         int   = 20      periodo de Bollinger
  bb_std            float = 2.0     desvíos estándar de Bollinger
  ema_fast          int   = 3       EMA rápida (cruce de corto plazo)
  ema_slow          int   = 8       EMA lenta  (cruce de corto plazo)
  drop_window       int   = 24      velas para calcular caída/subida
  rsi_buy_strong    float = 28.0    RSI fuerte para compra
  rsi_buy_weak      float = 35.0    RSI débil para compra
  rsi_sell_strong   float = 72.0    RSI fuerte para venta
  rsi_sell_weak     float = 62.0    RSI débil para venta
  drop_strong       float = -0.030  caída fuerte (-3%)
  drop_weak         float = -0.018  caída moderada (-1.8%)
  rise_strong       float = 0.030   subida fuerte (+3%)
  rise_weak         float = 0.018   subida moderada (+1.8%)
  low_rej_strong    float = 0.60    mecha inferior fuerte
  low_rej_weak      float = 0.40    mecha inferior moderada
  high_rej_strong   float = 0.60    mecha superior fuerte
  high_rej_weak     float = 0.40    mecha superior moderada
  puntos_buy_min    int   = 4       puntos mínimos para comprar
  puntos_sell_min   int   = 4       puntos mínimos para vender
  stop_loss_pct     float = 0.035   stop-loss del 3.5% por posición
  take_profit_pct   float = 0.025   take-profit parcial del 2.5%
  warmup_velas      int   = 50      velas de warmup antes de operar

Compatibilidad
───────────────
Implementa BaseStrategy — funciona con todos los runners existentes
(backtest_*.py, live_*.py) sin modificar ningún otro archivo.

El backtest de referencia se corre con:
    python backtest_mean_reversion.py

Para el runner live, copiar live_local_reversal.py y reemplazar
LocalReversalStrategy por MeanReversionStrategy.
No requiere entrenamiento ni cache — todos los cálculos son en línea.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

import numpy as np

from actors.price_feed import Candle
from actors.wallet import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger import get_logger

log = get_logger("mean_reversion")


# ══════════════════════════════════════════════════════════════════════════════
# CÁLCULOS TÉCNICOS (funciones puras, sin estado)
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(closes: np.ndarray, period: int) -> float:
    """
    RSI de Wilder sobre el array de closes.
    Retorna el RSI de la última vela.
    Requiere al menos period+1 valores.
    """
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    gains  = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = gains.mean()
    avg_loss = losses.mean()
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return float(100.0 - 100.0 / (1.0 + rs))


def _ema(closes: np.ndarray, period: int) -> float:
    """EMA sobre el array de closes. Retorna la EMA de la última vela."""
    if len(closes) < period:
        return float(closes[-1])
    k = 2.0 / (period + 1)
    ema = closes[-period]
    for c in closes[-period + 1:]:
        ema = c * k + ema * (1 - k)
    return float(ema)


def _bollinger(closes: np.ndarray, period: int, n_std: float) -> Tuple[float, float, float]:
    """
    Bandas de Bollinger. Retorna (lower, middle, upper).
    Requiere al menos period valores.
    """
    if len(closes) < period:
        c = float(closes[-1])
        return c, c, c
    window = closes[-period:]
    mid    = window.mean()
    std    = window.std(ddof=1)
    return float(mid - n_std * std), float(mid), float(mid + n_std * std)


def _drop_from_recent_high(closes: np.ndarray, window: int) -> float:
    """
    Variación del precio actual respecto al máximo de las últimas `window` velas.
    Retorna valor negativo si el precio bajó (ej. -0.03 = -3%).
    """
    if len(closes) < 2:
        return 0.0
    recent = closes[-min(window, len(closes)):]
    high   = recent.max()
    if high == 0:
        return 0.0
    return float((closes[-1] - high) / high)


def _rise_from_recent_low(closes: np.ndarray, window: int) -> float:
    """
    Variación del precio actual respecto al mínimo de las últimas `window` velas.
    Retorna valor positivo si el precio subió (ej. 0.04 = +4%).
    """
    if len(closes) < 2:
        return 0.0
    recent = closes[-min(window, len(closes)):]
    low    = recent.min()
    if low == 0:
        return 0.0
    return float((closes[-1] - low) / low)


def _low_rejection(candle: Candle) -> float:
    """
    Fracción de la mecha inferior: (close - low) / (high - low).
    Cercano a 1.0 = vela que rechazó fuertemente el bajo (señal alcista).
    """
    rng = candle.high - candle.low
    if rng < 1e-9:
        return 0.5
    return float((candle.close - candle.low) / rng)


def _high_rejection(candle: Candle) -> float:
    """
    Fracción de la mecha superior: (high - close) / (high - low).
    Cercano a 1.0 = vela que rechazó fuertemente el alto (señal bajista).
    """
    rng = candle.high - candle.low
    if rng < 1e-9:
        return 0.5
    return float((candle.high - candle.close) / rng)


# ══════════════════════════════════════════════════════════════════════════════
# ESTADO DE INDICADORES POR VELA
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class _Indicators:
    rsi:           float
    drop24:        float   # negativo = caída
    rise24:        float   # positivo = subida
    bb_lower:      float
    bb_upper:      float
    bb_mid:        float
    ema_fast:      float
    ema_slow:      float
    low_rej:       float
    high_rej:      float
    close:         float


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA
# ══════════════════════════════════════════════════════════════════════════════

class MeanReversionStrategy(BaseStrategy):
    """
    Estrategia de reversión a la media usando indicadores técnicos
    calibrados sobre los bottoms/tops reales del período 2021-2022.

    No requiere entrenamiento ni cache.
    Opera en tiempo real sobre cualquier PriceFeed compatible.
    """

    # ── Parámetros por defecto ────────────────────────────────────────────────
    DEFAULT_RSI_PERIOD      = 14
    DEFAULT_BB_PERIOD       = 20
    DEFAULT_BB_STD          = 2.0
    DEFAULT_EMA_FAST        = 3
    DEFAULT_EMA_SLOW        = 8
    DEFAULT_DROP_WINDOW     = 24
    DEFAULT_RSI_BUY_STRONG  = 28.0
    DEFAULT_RSI_BUY_WEAK    = 35.0
    DEFAULT_RSI_SELL_STRONG = 72.0
    DEFAULT_RSI_SELL_WEAK   = 62.0
    DEFAULT_DROP_STRONG     = -0.030
    DEFAULT_DROP_WEAK       = -0.018
    DEFAULT_RISE_STRONG     = 0.030
    DEFAULT_RISE_WEAK       = 0.018
    DEFAULT_LOW_REJ_STRONG  = 0.60
    DEFAULT_LOW_REJ_WEAK    = 0.40
    DEFAULT_HIGH_REJ_STRONG = 0.60
    DEFAULT_HIGH_REJ_WEAK   = 0.40
    DEFAULT_PUNTOS_BUY_MIN  = 4
    DEFAULT_PUNTOS_SELL_MIN = 4
    DEFAULT_STOP_LOSS_PCT   = 0.035
    DEFAULT_TAKE_PROFIT_PCT = 0.025
    DEFAULT_WARMUP_VELAS    = 50

    def __init__(
        self,
        rsi_period:       int   = DEFAULT_RSI_PERIOD,
        bb_period:        int   = DEFAULT_BB_PERIOD,
        bb_std:           float = DEFAULT_BB_STD,
        ema_fast:         int   = DEFAULT_EMA_FAST,
        ema_slow:         int   = DEFAULT_EMA_SLOW,
        drop_window:      int   = DEFAULT_DROP_WINDOW,
        rsi_buy_strong:   float = DEFAULT_RSI_BUY_STRONG,
        rsi_buy_weak:     float = DEFAULT_RSI_BUY_WEAK,
        rsi_sell_strong:  float = DEFAULT_RSI_SELL_STRONG,
        rsi_sell_weak:    float = DEFAULT_RSI_SELL_WEAK,
        drop_strong:      float = DEFAULT_DROP_STRONG,
        drop_weak:        float = DEFAULT_DROP_WEAK,
        rise_strong:      float = DEFAULT_RISE_STRONG,
        rise_weak:        float = DEFAULT_RISE_WEAK,
        low_rej_strong:   float = DEFAULT_LOW_REJ_STRONG,
        low_rej_weak:     float = DEFAULT_LOW_REJ_WEAK,
        high_rej_strong:  float = DEFAULT_HIGH_REJ_STRONG,
        high_rej_weak:    float = DEFAULT_HIGH_REJ_WEAK,
        puntos_buy_min:   int   = DEFAULT_PUNTOS_BUY_MIN,
        puntos_sell_min:  int   = DEFAULT_PUNTOS_SELL_MIN,
        stop_loss_pct:    float = DEFAULT_STOP_LOSS_PCT,
        take_profit_pct:  float = DEFAULT_TAKE_PROFIT_PCT,
        warmup_velas:     int   = DEFAULT_WARMUP_VELAS,
    ) -> None:
        super().__init__(name="MeanReversion-Tecnico")

        self.rsi_period      = rsi_period
        self.bb_period       = bb_period
        self.bb_std          = bb_std
        self.ema_fast        = ema_fast
        self.ema_slow        = ema_slow
        self.drop_window     = drop_window
        self.rsi_buy_strong  = rsi_buy_strong
        self.rsi_buy_weak    = rsi_buy_weak
        self.rsi_sell_strong = rsi_sell_strong
        self.rsi_sell_weak   = rsi_sell_weak
        self.drop_strong     = drop_strong
        self.drop_weak       = drop_weak
        self.rise_strong     = rise_strong
        self.rise_weak       = rise_weak
        self.low_rej_strong  = low_rej_strong
        self.low_rej_weak    = low_rej_weak
        self.high_rej_strong = high_rej_strong
        self.high_rej_weak   = high_rej_weak
        self.puntos_buy_min  = puntos_buy_min
        self.puntos_sell_min = puntos_sell_min
        self.stop_loss_pct   = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        self.warmup_velas    = warmup_velas

        # Buffer circular de closes (solo necesitamos closes para los indicadores)
        # Tamaño = max(rsi_period+1, bb_period, drop_window, ema_slow) + margen
        self._buf_size  = max(rsi_period + 2, bb_period, drop_window, ema_slow) + 10
        self._closes:   Deque[float] = deque(maxlen=self._buf_size)
        self._candles:  Deque[Candle] = deque(maxlen=4)  # últimas 4 velas para EMA cruce

        # Historial de indicadores para detectar cruce EMA
        self._prev_ema_fast: Optional[float] = None
        self._prev_ema_slow: Optional[float] = None

        log.info(
            "MeanReversionStrategy configurada",
            rsi_buy=f"{rsi_buy_strong}/{rsi_buy_weak}",
            rsi_sell=f"{rsi_sell_strong}/{rsi_sell_weak}",
            drop_strong=f"{drop_strong*100:.1f}%",
            rise_strong=f"{rise_strong*100:.1f}%",
            puntos_buy=puntos_buy_min,
            puntos_sell=puntos_sell_min,
            stop_loss=f"{stop_loss_pct*100:.1f}%" if stop_loss_pct > 0.0 else "desactivado",
        )

    # ══════════════════════════════════════════════════════════════════════════
    # INTERFAZ BaseStrategy
    # ══════════════════════════════════════════════════════════════════════════

    def on_start(self, wallet: Wallet, **kwargs) -> None:
        self._closes.clear()
        self._candles.clear()
        self._prev_ema_fast = None
        self._prev_ema_slow = None
        log.info("MeanReversionStrategy iniciada")

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        """
        Procesa cada vela nueva:
          1. Actualiza el buffer de closes.
          2. Calcula indicadores.
          3. Evalúa stop-loss (tiene prioridad sobre el resto).
          4. Evalúa señal de venta (puntaje).
          5. Evalúa señal de compra (puntaje).
        """
        # Actualizar buffers
        self._closes.append(candle.close)
        self._candles.append(candle)

        # Warmup: no operar hasta tener suficiente historia
        if self.candles_seen < self.warmup_velas:
            return HOLD

        closes = np.array(self._closes, dtype=np.float64)
        ind    = self._calc_indicators(candle, closes)

        # ── 1. Stop-loss ──────────────────────────────────────────────────────
        stop_signal = self._check_stop_loss(candle, wallet)
        if stop_signal is not None:
            return stop_signal

        # ── 2. Señal de venta (prioridad sobre compra) ────────────────────────
        if wallet.positions_count > 0:
            sell_signal = self._eval_sell(candle, ind, wallet)
            if sell_signal is not None:
                return sell_signal

        # ── 3. Señal de compra ────────────────────────────────────────────────
        buy_signal = self._eval_buy(candle, ind, wallet)
        if buy_signal is not None:
            return buy_signal

        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info(
            "MeanReversionStrategy detenida",
            velas_procesadas=self.candles_seen,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # CÁLCULO DE INDICADORES
    # ══════════════════════════════════════════════════════════════════════════

    def _calc_indicators(self, candle: Candle, closes: np.ndarray) -> _Indicators:
        rsi  = _rsi(closes, self.rsi_period)
        bb_l, bb_m, bb_u = _bollinger(closes, self.bb_period, self.bb_std)
        drop = _drop_from_recent_high(closes, self.drop_window)
        rise = _rise_from_recent_low(closes, self.drop_window)
        ef   = _ema(closes, self.ema_fast)
        es   = _ema(closes, self.ema_slow)
        lr   = _low_rejection(candle)
        hr   = _high_rejection(candle)
        return _Indicators(
            rsi=rsi, drop24=drop, rise24=rise,
            bb_lower=bb_l, bb_upper=bb_u, bb_mid=bb_m,
            ema_fast=ef, ema_slow=es,
            low_rej=lr, high_rej=hr,
            close=candle.close,
        )

    # ══════════════════════════════════════════════════════════════════════════
    # EVALUACIÓN DE SEÑALES
    # ══════════════════════════════════════════════════════════════════════════

    def _eval_buy(
        self,
        candle: Candle,
        ind:    _Indicators,
        wallet: Wallet,
    ) -> Optional[Signal]:
        """
        Sistema de puntos para la señal de compra.
        Cada condición aporta puntos según su confiabilidad.
        """
        pts   = 0
        razones: list[str] = []

        # RSI (mayor peso — indicador más robusto)
        if ind.rsi < self.rsi_buy_strong:
            pts += 3
            razones.append(f"RSI={ind.rsi:.1f}<{self.rsi_buy_strong}")
        elif ind.rsi < self.rsi_buy_weak:
            pts += 2
            razones.append(f"RSI={ind.rsi:.1f}<{self.rsi_buy_weak}")

        # Caída reciente (calibrada en el irreal: -2% cubre 82% de bottoms)
        if ind.drop24 < self.drop_strong:
            pts += 2
            razones.append(f"drop={ind.drop24*100:.1f}%")
        elif ind.drop24 < self.drop_weak:
            pts += 1
            razones.append(f"drop={ind.drop24*100:.1f}%")

        # Bollinger inferior
        if ind.close < ind.bb_lower:
            pts += 2
            razones.append("bajo_BB")
        elif ind.close < ind.bb_lower * 1.005:
            pts += 1
            razones.append("cerca_BB_inf")

        # Mecha inferior (rechazo del bajo)
        if ind.low_rej > self.low_rej_strong:
            pts += 2
            razones.append(f"low_rej={ind.low_rej:.2f}")
        elif ind.low_rej > self.low_rej_weak:
            pts += 1
            razones.append(f"low_rej={ind.low_rej:.2f}")

        # Cruce EMA rápida sobre lenta (inicio de rebote)
        if (self._prev_ema_fast is not None and self._prev_ema_slow is not None
                and self._prev_ema_fast <= self._prev_ema_slow
                and ind.ema_fast > ind.ema_slow):
            pts += 1
            razones.append("cruce_EMA_alcista")

        # Actualizar EMAs previas
        self._prev_ema_fast = ind.ema_fast
        self._prev_ema_slow = ind.ema_slow

        if pts < self.puntos_buy_min:
            return None

        razon = f"buy_pts={pts} [{', '.join(razones)}]"
        log.debug("señal BUY", puntos=pts, razon=razon, rsi=f"{ind.rsi:.1f}",
                  drop=f"{ind.drop24*100:.1f}%", close=f"{candle.close:.0f}")
        return Signal(
            side   = SignalSide.BUY,
            price  = candle.close,
            reason = razon,
            score  = min(1.0, pts / 10.0),
        )

    def _eval_sell(
        self,
        candle: Candle,
        ind:    _Indicators,
        wallet: Wallet,
    ) -> Optional[Signal]:
        """
        Sistema de puntos para la señal de venta.
        También evalúa take-profit si la posición está en ganancia.
        """
        pts   = 0
        razones: list[str] = []

        # RSI
        if ind.rsi > self.rsi_sell_strong:
            pts += 3
            razones.append(f"RSI={ind.rsi:.1f}>{self.rsi_sell_strong}")
        elif ind.rsi > self.rsi_sell_weak:
            pts += 2
            razones.append(f"RSI={ind.rsi:.1f}>{self.rsi_sell_weak}")

        # Subida reciente (calibrada en el irreal: +2% cubre 87% de tops)
        if ind.rise24 > self.rise_strong:
            pts += 2
            razones.append(f"rise={ind.rise24*100:.1f}%")
        elif ind.rise24 > self.rise_weak:
            pts += 1
            razones.append(f"rise={ind.rise24*100:.1f}%")

        # Bollinger superior
        if ind.close > ind.bb_upper:
            pts += 2
            razones.append("sobre_BB")
        elif ind.close > ind.bb_upper * 0.995:
            pts += 1
            razones.append("cerca_BB_sup")

        # Mecha superior (rechazo del alto)
        if ind.high_rej > self.high_rej_strong:
            pts += 2
            razones.append(f"high_rej={ind.high_rej:.2f}")
        elif ind.high_rej > self.high_rej_weak:
            pts += 1
            razones.append(f"high_rej={ind.high_rej:.2f}")

        # Take-profit: añade 1 punto extra si ya ganamos lo suficiente
        avg_entry = wallet.precio_promedio_posiciones()
        if avg_entry > 0:
            gain = (candle.close - avg_entry) / avg_entry
            if gain > self.take_profit_pct:
                pts += 1
                razones.append(f"tp={gain*100:.1f}%")

        if pts < self.puntos_sell_min:
            return None

        razon = f"sell_pts={pts} [{', '.join(razones)}]"
        log.debug("señal SELL", puntos=pts, razon=razon, rsi=f"{ind.rsi:.1f}",
                  rise=f"{ind.rise24*100:.1f}%", close=f"{candle.close:.0f}")
        return Signal(
            side   = SignalSide.SELL,
            price  = candle.close,
            reason = razon,
            score  = min(1.0, pts / 10.0),
        )

    def _check_stop_loss(
        self,
        candle: Candle,
        wallet: Wallet,
    ) -> Optional[Signal]:
        """
        Stop-loss por posición: si el precio actual está más de
        STOP_LOSS_PCT por debajo del precio de entrada de la posición
        más antigua (FIFO), emite señal de venta de emergencia.

        La señal tiene score=0 para diferenciarla de ventas normales
        en los logs y análisis posteriores.

        Si stop_loss_pct <= 0.0 el stop-loss está desactivado y retorna None
        inmediatamente, sin evaluar posiciones.
        """
        if self.stop_loss_pct <= 0.0:
            return None

        positions = wallet.get_positions()
        if not positions:
            return None
        oldest = positions[0]
        if oldest.entry_price <= 0:
            return None
        loss_pct = (candle.close - oldest.entry_price) / oldest.entry_price
        if loss_pct < -self.stop_loss_pct:
            razon = (f"stop_loss={loss_pct*100:.2f}% "
                     f"(entry={oldest.entry_price:.0f} "
                     f"current={candle.close:.0f})")
            log.info("STOP-LOSS activado", razon=razon)
            return Signal(
                side   = SignalSide.SELL,
                price  = candle.close,
                reason = razon,
                score  = 0.0,  # score=0 = stop-loss (no señal de modelo)
            )
        return None

    # ══════════════════════════════════════════════════════════════════════════
    # METADATA
    # ══════════════════════════════════════════════════════════════════════════

    def describe(self) -> dict:
        return {
            "estrategia":       self.name,
            "rsi_period":       self.rsi_period,
            "bb_period":        self.bb_period,
            "bb_std":           self.bb_std,
            "ema_fast":         self.ema_fast,
            "ema_slow":         self.ema_slow,
            "drop_window":      self.drop_window,
            "rsi_buy":          f"{self.rsi_buy_strong}/{self.rsi_buy_weak}",
            "rsi_sell":         f"{self.rsi_sell_strong}/{self.rsi_sell_weak}",
            "drop_strong":      self.drop_strong,
            "drop_weak":        self.drop_weak,
            "rise_strong":      self.rise_strong,
            "rise_weak":        self.rise_weak,
            "low_rej_strong":   self.low_rej_strong,
            "high_rej_strong":  self.high_rej_strong,
            "puntos_buy_min":   self.puntos_buy_min,
            "puntos_sell_min":  self.puntos_sell_min,
            "stop_loss_pct":    self.stop_loss_pct,
            "take_profit_pct":  self.take_profit_pct,
            "warmup_velas":     self.warmup_velas,
            # Campos esperados por Graficador.py
            "rsi_length":        self.rsi_period,
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "N":                 self.drop_window,
            "guardia_compra":    True,
            "guardia_venta":     True,
        }
