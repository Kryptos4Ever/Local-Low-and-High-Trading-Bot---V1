"""
strategies/zigzag_reversal.py — ZigZag Reversal Strategy
═══════════════════════════════════════════════════════════
Estrategia de reversión calibrada empíricamente sobre los turning points
detectados por el ZigZag 5% en el período 2021-11-10 → 2022-11-22.

Fundamento estadístico (99 BOTTOMs / 98 TOPs analizados)
──────────────────────────────────────────────────────────
Las condiciones se seleccionaron por su poder discriminativo (separabilidad
entre la distribución de BOTTOMs y TOPs), usando exclusivamente features
con CV bajo (consistencia alta):

  Feature               CV_Bot  CV_Top  Discriminación
  ──────────────────────────────────────────────────────
  taker_ratio_mean      0.025   0.025   pequeña pero robusta (más confiable)
  taker_last5_vs_avg    0.030   0.027   pequeña pero robusta
  rsi_at_end            0.286   0.192   +71.7pp  (la más discriminativa)
  pct_bearish_candles   0.182   0.313   +64.6pp
  price_vs_ma20_pct     0.892   1.337   +74.7pp
  below/above_both_mas  binaria          +81.7pp / +74.5pp
  ma20_slope_last5      1.053   3.824   +65.4pp
  rsi_slope_last5       1.091   1.632   +75.6pp
  price_slope           1.593   1.727   +99.0pp (100% / 99% cobertura)

Umbrales derivados de los percentiles del CSV (no de la media):
  BUY:  RSI<40 cubre 78% de BOTTOMs con 6% falsos positivos
        price_slope<0 cubre 100% de BOTTOMs con 1% falsos positivos
        below_both_mas=1 cubre 90% de BOTTOMs con 8% falsos positivos
  SELL: price_slope>0 cubre 99% de TOPs con 0% falsos positivos
        rsi_slope>0 cubre 88% de TOPs con 12% falsos positivos
        above_both_mas=1 cubre 77% de TOPs con 2% falsos positivos

Sistema de puntos
──────────────────
En lugar de AND rígido (muy restrictivo) o OR puro (muchas señales falsas),
se usa un sistema de puntos donde cada condición aporta según su
discriminabilidad medida:

  COMPRA  si puntos_buy  >= MIN_PUNTOS_BUY   (default 5)
  VENTA   si puntos_sell >= MIN_PUNTOS_SELL  (default 5)

Ventana de cómputo
───────────────────
Todos los indicadores se calculan sobre los últimas WINDOW velas
del buffer interno (default 24 velas = 24h para velas de 1h).
Esto replica el "segmento hasta el turning point" que usó el análisis.

La taker_ratio_slope y rsi_slope se calculan sobre las últimas 5 velas
(LAST_N = 5) para consistencia con las features del CSV.

Operación en producción
────────────────────────
· No requiere entrenamiento ni cache.
· on_start() no necesita feed ni fechas — inicializa solo el buffer.
· Cada on_candle() actualiza el buffer en O(1) y calcula indicadores
  en O(WINDOW), completamente en tiempo real.
· Compatible con BinanceWSFeed (velas de 1h) sin modificaciones.

Parámetros configurables
─────────────────────────
Todos los parámetros tienen defaults calibrados empíricamente.
Para ajustar el agresividad: bajar MIN_PUNTOS_* para más operaciones,
subir para menos pero más selectivas.

Para grid search rápido ver backtest_zigzag_reversal.py --grid.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional, Tuple

from actors.price_feed        import Candle
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("zigzag_reversal")


# ═══════════════════════════════════════════════════════════════════
# CÁLCULOS TÉCNICOS (funciones puras)
# ═══════════════════════════════════════════════════════════════════

def _rsi(closes: List[float], period: int = 14) -> Optional[float]:
    """RSI de Wilder. Requiere al menos period+1 valores. Retorna None si insuficientes."""
    n = len(closes)
    if n < period + 1:
        return None
    deltas = [closes[i] - closes[i - 1] for i in range(1, n)]
    gains  = [max(d, 0.0) for d in deltas[-period:]]
    losses = [max(-d, 0.0) for d in deltas[-period:]]
    avg_g  = sum(gains)  / period
    avg_l  = sum(losses) / period
    if avg_l == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_g / avg_l)


def _sma(values: List[float], period: int) -> Optional[float]:
    """SMA simple. Retorna None si no hay suficientes valores."""
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _linear_slope_norm(values: List[float]) -> float:
    """
    Pendiente lineal normalizada por el valor medio.
    Retorna %/vela. Igual que linear_slope_norm() en analyze_turning_points.py.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    if y_mean == 0:
        return 0.0
    numer = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    denom = sum((i - x_mean) ** 2                   for i in range(n))
    if denom == 0:
        return 0.0
    return (numer / denom) / abs(y_mean)


@dataclass(slots=True)
class _State:
    """Estado de todos los indicadores para la vela actual."""
    rsi:               Optional[float]  # RSI(14) al cierre de la vela
    rsi_slope:         float            # pendiente RSI últimas LAST_N velas
    price_vs_ma20:     Optional[float]  # (close - MA20) / MA20 * 100
    price_vs_ma50:     Optional[float]  # (close - MA50) / MA50 * 100
    ma20_slope:        float            # pendiente MA20 últimas LAST_N velas
    below_both_mas:    bool             # close < MA20 y close < MA50
    above_both_mas:    bool             # close > MA20 y close > MA50
    taker_ratio:       Optional[float]  # taker_buy_base_vol / volume (vela actual)
    taker_mean:        Optional[float]  # media de taker_ratio en WINDOW velas
    taker_last5:       Optional[float]  # taker_last5_vs_avg (ratio vs media del segmento)
    taker_slope:       float            # pendiente taker LAST_N velas
    lower_wick:        float            # (close - low) / (high - low)
    upper_wick:        float            # (high - close) / (high - low)
    pct_bearish:       float            # % velas bajistas en WINDOW velas
    price_slope:       float            # pendiente precio WINDOW velas
    vol_last5_vs_avg:  Optional[float]  # vol últimas 5 vs media del WINDOW


# ═══════════════════════════════════════════════════════════════════
# ESTRATEGIA
# ═══════════════════════════════════════════════════════════════════

class ZigZagReversalStrategy(BaseStrategy):
    """
    Detecta reversiones locales usando los mismos indicadores que el análisis
    de turning points, calibrados sobre sus percentiles reales.

    No requiere entrenamiento. Opera en tiempo real sobre cualquier PriceFeed.
    """

    # ── Parámetros de indicadores ─────────────────────────────────
    DEFAULT_RSI_PERIOD      = 14
    DEFAULT_MA_SHORT        = 20
    DEFAULT_MA_LONG         = 50
    DEFAULT_WINDOW          = 24    # velas del "segmento" para features
    DEFAULT_LAST_N          = 5     # últimas N velas para slopes y ratios

    # ── Umbrales BUY (calibrados sobre percentiles de BOTTOMs) ────
    # RSI: p25 BOTTOMs = 27.6, p75 BOTTOMs = 38.4
    DEFAULT_RSI_BUY_STRONG  = 30.0  # cubre 35% BOTTOMs, 2% TOPs  → 3 pts
    DEFAULT_RSI_BUY_WEAK    = 40.0  # cubre 78% BOTTOMs, 6% TOPs  → 2 pts
    # price_vs_ma20: p50 BOTTOMs = -2.47%, p25 = -3.55%
    DEFAULT_MA20_BUY_STRONG = -3.5  # cubre ~55% BOTTOMs, ~5% TOPs → 2 pts
    DEFAULT_MA20_BUY_WEAK   = -1.0  # cubre 82% BOTTOMs, 7% TOPs  → 1 pt
    # Binarias con alta discriminabilidad
    # below_both_mas: 90% BOTTOMs / 8% TOPs → 2 pts
    # rsi_slope < 0: 88% BOTTOMs / 12% TOPs → 1 pt
    # taker_slope < 0: 79% BOTTOMs / 31% TOPs → 1 pt
    # pct_bearish > 55%: 68% BOTTOMs / 3% TOPs → 1 pt
    # ma20_slope < 0: 91% BOTTOMs / 26% TOPs → 1 pt
    # lower_wick > 0.28: 58% BOTTOMs / 43% TOPs → 1 pt (señal de reversión inmediata)
    # vol_last5 > 1.2: 49% BOTTOMs / 40% TOPs → 0.5 pt (señal débil)

    # ── Umbrales SELL (calibrados sobre percentiles de TOPs) ──────
    # RSI: p25 TOPs = 54.4, p50 = 63.8, p75 = 69.2
    DEFAULT_RSI_SELL_STRONG = 65.0  # cubre 43% TOPs, 0% BOTTOMs → 3 pts
    DEFAULT_RSI_SELL_WEAK   = 55.0  # cubre 74% TOPs, 1% BOTTOMs → 2 pts
    # price_vs_ma20: p50 TOPs = +1.85%, p75 = +2.57%
    DEFAULT_MA20_SELL_STRONG= 2.5   # cubre 37% TOPs, 2% BOTTOMs → 2 pts
    DEFAULT_MA20_SELL_WEAK  = 1.0   # cubre 75% TOPs, 3% BOTTOMs → 1 pt
    # Binarias SELL
    # above_both_mas: 77% TOPs / 2% BOTTOMs → 2 pts
    # rsi_slope > 0: 88% TOPs / 12% BOTTOMs → 1 pt
    # taker_slope > 0: 69% TOPs / 21% BOTTOMs → 1 pt
    # pct_bullish > 55%: 64% TOPs / 3% BOTTOMs → 1 pt
    # ma20_slope > 0: 75% TOPs / 9% BOTTOMs → 1 pt
    # upper_wick > 0.29: 64% TOPs / 37% BOTTOMs → 1 pt

    # ── Puntaje mínimo ────────────────────────────────────────────
    DEFAULT_MIN_PUNTOS_BUY  = 5
    DEFAULT_MIN_PUNTOS_SELL = 5

    # ── Gestión de riesgo ─────────────────────────────────────────
    DEFAULT_STOP_LOSS_PCT   = 0.04   # -4% por posición (p90 move_pct bottoms = 16.8%)
    DEFAULT_WARMUP          = 55     # max(MA_LONG + RSI_PERIOD + LAST_N + margen)

    def __init__(
        self,
        rsi_period:       int   = DEFAULT_RSI_PERIOD,
        ma_short:         int   = DEFAULT_MA_SHORT,
        ma_long:          int   = DEFAULT_MA_LONG,
        window:           int   = DEFAULT_WINDOW,
        last_n:           int   = DEFAULT_LAST_N,
        rsi_buy_strong:   float = DEFAULT_RSI_BUY_STRONG,
        rsi_buy_weak:     float = DEFAULT_RSI_BUY_WEAK,
        ma20_buy_strong:  float = DEFAULT_MA20_BUY_STRONG,
        ma20_buy_weak:    float = DEFAULT_MA20_BUY_WEAK,
        rsi_sell_strong:  float = DEFAULT_RSI_SELL_STRONG,
        rsi_sell_weak:    float = DEFAULT_RSI_SELL_WEAK,
        ma20_sell_strong: float = DEFAULT_MA20_SELL_STRONG,
        ma20_sell_weak:   float = DEFAULT_MA20_SELL_WEAK,
        min_puntos_buy:   int   = DEFAULT_MIN_PUNTOS_BUY,
        min_puntos_sell:  int   = DEFAULT_MIN_PUNTOS_SELL,
        stop_loss_pct:    float = DEFAULT_STOP_LOSS_PCT,
        warmup:           int   = DEFAULT_WARMUP,
    ) -> None:
        super().__init__(name="ZigZagReversal-Calibrado")

        self.rsi_period       = rsi_period
        self.ma_short         = ma_short
        self.ma_long          = ma_long
        self.window           = window
        self.last_n           = last_n
        self.rsi_buy_strong   = rsi_buy_strong
        self.rsi_buy_weak     = rsi_buy_weak
        self.ma20_buy_strong  = ma20_buy_strong
        self.ma20_buy_weak    = ma20_buy_weak
        self.rsi_sell_strong  = rsi_sell_strong
        self.rsi_sell_weak    = rsi_sell_weak
        self.ma20_sell_strong = ma20_sell_strong
        self.ma20_sell_weak   = ma20_sell_weak
        self.min_puntos_buy   = min_puntos_buy
        self.min_puntos_sell  = min_puntos_sell
        self.stop_loss_pct    = stop_loss_pct
        self.warmup           = warmup

        # Tamaño del buffer: lo suficiente para todos los indicadores
        buf = max(rsi_period + last_n + 2, ma_long + last_n + 2, window + 2)
        self._closes:  Deque[float]          = deque(maxlen=buf)
        self._takers:  Deque[Optional[float]]= deque(maxlen=buf)
        self._volumes: Deque[float]          = deque(maxlen=buf)
        self._rsi_buf: Deque[Optional[float]]= deque(maxlen=last_n + 2)
        self._ma20_buf:Deque[Optional[float]]= deque(maxlen=last_n + 2)

        log.info(
            "ZigZagReversalStrategy configurada",
            min_buy=min_puntos_buy, min_sell=min_puntos_sell,
            rsi_buy=f"{rsi_buy_strong}/{rsi_buy_weak}",
            rsi_sell=f"{rsi_sell_strong}/{rsi_sell_weak}",
            stop_loss=f"{stop_loss_pct*100:.1f}%",
        )

    # ══════════════════════════════════════════════════════════════
    # INTERFAZ BaseStrategy
    # ══════════════════════════════════════════════════════════════

    def on_start(self, wallet: Wallet, **kwargs) -> None:
        self._closes.clear()
        self._takers.clear()
        self._volumes.clear()
        self._rsi_buf.clear()
        self._ma20_buf.clear()
        log.info("ZigZagReversalStrategy iniciada")

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        # 1. Actualizar buffers
        self._closes.append(candle.close)
        self._volumes.append(candle.volume)
        taker = (candle.taker_buy_base_vol / candle.volume
                 if candle.taker_buy_base_vol is not None and candle.volume > 0
                 else None)
        self._takers.append(taker)

        # 2. Calcular RSI y MA20 para sus propios buffers de slope
        closes_list = list(self._closes)
        rsi_val  = _rsi(closes_list, self.rsi_period)
        ma20_val = _sma(closes_list, self.ma_short)
        self._rsi_buf.append(rsi_val)
        self._ma20_buf.append(ma20_val)

        # 3. Warmup
        if self.candles_seen < self.warmup:
            return HOLD

        # 4. Calcular todos los indicadores
        state = self._compute_state(candle, closes_list)

        # 5. Stop-loss (máxima prioridad)
        stop = self._check_stop_loss(candle, wallet)
        if stop is not None:
            return stop

        # 6. Señal de venta (prioridad sobre compra — evita abrir y cerrar en la misma vela)
        if wallet.positions_count > 0:
            sell = self._eval_sell(candle, state, wallet)
            if sell is not None:
                return sell

        # 7. Señal de compra
        buy = self._eval_buy(candle, state)
        if buy is not None:
            return buy

        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info("ZigZagReversalStrategy detenida", velas=self.candles_seen)

    # ══════════════════════════════════════════════════════════════
    # CÓMPUTO DE ESTADO
    # ══════════════════════════════════════════════════════════════

    def _compute_state(self, candle: Candle, closes: List[float]) -> _State:
        n = len(closes)

        # RSI
        rsi_val = list(self._rsi_buf)[-1] if self._rsi_buf else None

        # RSI slope sobre últimas LAST_N velas con RSI calculado
        rsi_vals = [v for v in self._rsi_buf if v is not None]
        rsi_slope = _linear_slope_norm(rsi_vals[-self.last_n:]) if len(rsi_vals) >= 2 else 0.0

        # MAs
        ma20_val = _sma(closes, self.ma_short)
        ma50_val = _sma(closes, self.ma_long)

        # price vs MAs
        price_vs_ma20 = None
        price_vs_ma50 = None
        if ma20_val:
            price_vs_ma20 = (candle.close - ma20_val) / ma20_val * 100
        if ma50_val:
            price_vs_ma50 = (candle.close - ma50_val) / ma50_val * 100

        # MA20 slope
        ma20_vals = [v for v in self._ma20_buf if v is not None]
        ma20_slope = _linear_slope_norm(ma20_vals[-self.last_n:]) if len(ma20_vals) >= 2 else 0.0

        # below/above ambas MAs
        below = bool(ma20_val and ma50_val
                     and candle.close < ma20_val and candle.close < ma50_val)
        above = bool(ma20_val and ma50_val
                     and candle.close > ma20_val and candle.close > ma50_val)

        # Taker ratio
        taker_list = [v for v in self._takers if v is not None]
        win_takers = taker_list[-self.window:] if taker_list else []
        taker_mean = sum(win_takers) / len(win_takers) if win_takers else None

        # taker_last5_vs_avg
        last5_takers = [v for v in taker_list[-self.last_n:]]
        taker_last5 = None
        if last5_takers and taker_mean and taker_mean > 0:
            taker_last5 = sum(last5_takers) / len(last5_takers) / taker_mean

        # taker slope
        taker_slope = (_linear_slope_norm(taker_list[-self.last_n:])
                       if len(taker_list) >= 2 else 0.0)

        # Mechas
        rng = candle.high - candle.low
        if rng > 0:
            lower_wick = (candle.close - candle.low)  / rng
            upper_wick = (candle.high  - candle.close) / rng
        else:
            lower_wick = upper_wick = 0.5

        # % velas bajistas en ventana
        win_closes = closes[-self.window:]
        bearish = 0
        for i in range(1, len(win_closes)):
            if win_closes[i] < win_closes[i - 1]:
                bearish += 1
        pct_bearish = bearish / max(1, len(win_closes) - 1) * 100

        # Price slope sobre la ventana completa
        price_slope = _linear_slope_norm(closes[-self.window:]) if n >= 2 else 0.0

        # vol_last5_vs_avg
        vols = list(self._volumes)
        win_vols = vols[-self.window:]
        vol_mean_win = sum(win_vols) / len(win_vols) if win_vols else None
        last5_vols   = vols[-self.last_n:]
        vol_last5 = None
        if last5_vols and vol_mean_win and vol_mean_win > 0:
            vol_last5 = sum(last5_vols) / len(last5_vols) / vol_mean_win

        return _State(
            rsi            = rsi_val,
            rsi_slope      = rsi_slope,
            price_vs_ma20  = price_vs_ma20,
            price_vs_ma50  = price_vs_ma50,
            ma20_slope     = ma20_slope,
            below_both_mas = below,
            above_both_mas = above,
            taker_ratio    = list(self._takers)[-1],
            taker_mean     = taker_mean,
            taker_last5    = taker_last5,
            taker_slope    = taker_slope,
            lower_wick     = lower_wick,
            upper_wick     = upper_wick,
            pct_bearish    = pct_bearish,
            price_slope    = price_slope,
            vol_last5_vs_avg = vol_last5,
        )

    # ══════════════════════════════════════════════════════════════
    # EVALUACIÓN DE SEÑALES
    # ══════════════════════════════════════════════════════════════

    def _eval_buy(self, candle: Candle, s: _State) -> Optional[Signal]:
        """
        Sistema de puntos para BUY.
        Puntos máximos posibles: 3+2+2+1+2+1+1+1+1+1 = 15
        Default mínimo: 5 puntos.

        Discriminabilidad medida por condición (del CSV):
          RSI fuerte (<30):      35% BOTTOMs, 2% TOPs   → 3 pts
          RSI débil (<40):       78% BOTTOMs, 6% TOPs   → 2 pts
          MA20 strong (<-3.5%):  55% BOTTOMs, 5% TOPs   → 2 pts
          MA20 weak (<-1%):      82% BOTTOMs, 7% TOPs   → 1 pt
          below_both_mas:        90% BOTTOMs, 8% TOPs   → 2 pts
          rsi_slope < 0:         88% BOTTOMs, 12% TOPs  → 1 pt
          taker_slope < 0:       79% BOTTOMs, 31% TOPs  → 1 pt
          pct_bearish > 55%:     68% BOTTOMs, 3% TOPs   → 1 pt
          ma20_slope < 0:        91% BOTTOMs, 26% TOPs  → 1 pt
          lower_wick > 0.28:     58% BOTTOMs, 43% TOPs  → 1 pt
        """
        pts     = 0
        reasons = []

        # RSI (mayor discriminabilidad → más puntos)
        if s.rsi is not None:
            if s.rsi < self.rsi_buy_strong:
                pts += 3
                reasons.append(f"RSI={s.rsi:.1f}<{self.rsi_buy_strong}")
            elif s.rsi < self.rsi_buy_weak:
                pts += 2
                reasons.append(f"RSI={s.rsi:.1f}<{self.rsi_buy_weak}")

        # Distancia a MA20 (segundo más discriminativo)
        if s.price_vs_ma20 is not None:
            if s.price_vs_ma20 < self.ma20_buy_strong:
                pts += 2
                reasons.append(f"ma20={s.price_vs_ma20:.2f}%")
            elif s.price_vs_ma20 < self.ma20_buy_weak:
                pts += 1
                reasons.append(f"ma20={s.price_vs_ma20:.2f}%")

        # Posición relativa a ambas MAs (binaria, alta discriminabilidad)
        if s.below_both_mas:
            pts += 2
            reasons.append("below_MAs")

        # RSI cayendo (88% de BOTTOMs)
        if s.rsi_slope < 0:
            pts += 1
            reasons.append(f"rsi_slope={s.rsi_slope:.4f}")

        # Taker ratio bajando (presión vendedora — agotamiento de la caída)
        if s.taker_slope < 0:
            pts += 1
            reasons.append(f"taker_slope={s.taker_slope:.4f}")

        # Mayoría de velas bajistas en el segmento
        if s.pct_bearish > 55.0:
            pts += 1
            reasons.append(f"bearish={s.pct_bearish:.0f}%")

        # MA20 cayendo (90% de BOTTOMs)
        if s.ma20_slope < 0:
            pts += 1
            reasons.append(f"ma20_slope={s.ma20_slope:.5f}")

        # Mecha inferior pronunciada (señal de reversión inmediata)
        if s.lower_wick > 0.28:
            pts += 1
            reasons.append(f"low_wick={s.lower_wick:.2f}")

        if pts < self.min_puntos_buy:
            return None

        reason = f"buy pts={pts}/{self.min_puntos_buy} [{', '.join(reasons)}]"
        log.debug("BUY signal", pts=pts, rsi=s.rsi, ma20=s.price_vs_ma20)
        return Signal(
            side   = SignalSide.BUY,
            price  = candle.close,
            reason = reason,
            score  = round(min(1.0, pts / 12.0), 3),
        )

    def _eval_sell(self, candle: Candle, s: _State, wallet: Wallet) -> Optional[Signal]:
        """
        Sistema de puntos para SELL.
        Puntos máximos posibles: 3+2+2+1+2+1+1+1+1+1 = 15
        Default mínimo: 5 puntos.

        Discriminabilidad medida por condición:
          RSI fuerte (>65):      43% TOPs, 0% BOTTOMs   → 3 pts
          RSI débil (>55):       74% TOPs, 1% BOTTOMs   → 2 pts
          MA20 strong (>2.5%):   37% TOPs, 2% BOTTOMs   → 2 pts
          MA20 weak (>1%):       75% TOPs, 3% BOTTOMs   → 1 pt
          above_both_mas:        77% TOPs, 2% BOTTOMs   → 2 pts
          rsi_slope > 0:         88% TOPs, 12% BOTTOMs  → 1 pt
          taker_slope > 0:       69% TOPs, 21% BOTTOMs  → 1 pt
          pct_bullish > 55%:     64% TOPs, 3% BOTTOMs   → 1 pt
          ma20_slope > 0:        75% TOPs, 9% BOTTOMs   → 1 pt
          upper_wick > 0.29:     64% TOPs, 37% BOTTOMs  → 1 pt
        """
        pts     = 0
        reasons = []

        # RSI
        if s.rsi is not None:
            if s.rsi > self.rsi_sell_strong:
                pts += 3
                reasons.append(f"RSI={s.rsi:.1f}>{self.rsi_sell_strong}")
            elif s.rsi > self.rsi_sell_weak:
                pts += 2
                reasons.append(f"RSI={s.rsi:.1f}>{self.rsi_sell_weak}")

        # Distancia a MA20
        if s.price_vs_ma20 is not None:
            if s.price_vs_ma20 > self.ma20_sell_strong:
                pts += 2
                reasons.append(f"ma20={s.price_vs_ma20:.2f}%")
            elif s.price_vs_ma20 > self.ma20_sell_weak:
                pts += 1
                reasons.append(f"ma20={s.price_vs_ma20:.2f}%")

        # Posición relativa a ambas MAs
        if s.above_both_mas:
            pts += 2
            reasons.append("above_MAs")

        # RSI subiendo
        if s.rsi_slope > 0:
            pts += 1
            reasons.append(f"rsi_slope={s.rsi_slope:.4f}")

        # Taker subiendo (presión compradora — agotamiento de la subida)
        if s.taker_slope > 0:
            pts += 1
            reasons.append(f"taker_slope={s.taker_slope:.4f}")

        # Mayoría de velas alcistas
        if s.pct_bearish < 45.0:  # = pct_bullish > 55%
            pts += 1
            reasons.append(f"bullish={100-s.pct_bearish:.0f}%")

        # MA20 subiendo
        if s.ma20_slope > 0:
            pts += 1
            reasons.append(f"ma20_slope={s.ma20_slope:.5f}")

        # Mecha superior pronunciada
        if s.upper_wick > 0.29:
            pts += 1
            reasons.append(f"up_wick={s.upper_wick:.2f}")

        # Bonus: ganancia desde entrada
        avg_entry = wallet.precio_promedio_posiciones()
        if avg_entry > 0:
            gain = (candle.close - avg_entry) / avg_entry
            if gain > 0.03:   # +3% desde entrada = p50 del move_pct
                pts += 1
                reasons.append(f"gain={gain*100:.1f}%")

        if pts < self.min_puntos_sell:
            return None

        reason = f"sell pts={pts}/{self.min_puntos_sell} [{', '.join(reasons)}]"
        log.debug("SELL signal", pts=pts, rsi=s.rsi, ma20=s.price_vs_ma20)
        return Signal(
            side   = SignalSide.SELL,
            price  = candle.close,
            reason = reason,
            score  = round(min(1.0, pts / 12.0), 3),
        )

    def _check_stop_loss(self, candle: Candle, wallet: Wallet) -> Optional[Signal]:
        """
        Stop-loss por posición FIFO más antigua.
        Umbral: -4% desde la entrada (p90 del move_pct = 16.8%, así que
        un stop del 4% corta las pérdidas antes de que lleguen a la magnitud
        de un giro completo de ZigZag).
        score=0.0 para identificar stops en el log de trades.
        """
        if self.stop_loss_pct <= 0:
            return None
        positions = wallet.get_positions()
        if not positions:
            return None
        oldest = positions[0]
        if oldest.entry_price <= 0:
            return None
        loss = (candle.close - oldest.entry_price) / oldest.entry_price
        if loss < -self.stop_loss_pct:
            razon = (f"stop_loss={loss*100:.2f}% "
                     f"entry={oldest.entry_price:.0f} now={candle.close:.0f}")
            log.info("STOP-LOSS", razon=razon)
            return Signal(
                side   = SignalSide.SELL,
                price  = candle.close,
                reason = razon,
                score  = 0.0,
            )
        return None

    # ══════════════════════════════════════════════════════════════
    # METADATA
    # ══════════════════════════════════════════════════════════════

    def describe(self) -> dict:
        return {
            "estrategia":        self.name,
            "rsi_period":        self.rsi_period,
            "ma_short":          self.ma_short,
            "ma_long":           self.ma_long,
            "window":            self.window,
            "last_n":            self.last_n,
            "rsi_buy":           f"{self.rsi_buy_strong}/{self.rsi_buy_weak}",
            "rsi_sell":          f"{self.rsi_sell_strong}/{self.rsi_sell_weak}",
            "ma20_buy":          f"{self.ma20_buy_strong}/{self.ma20_buy_weak}",
            "ma20_sell":         f"{self.ma20_sell_strong}/{self.ma20_sell_weak}",
            "min_puntos_buy":    self.min_puntos_buy,
            "min_puntos_sell":   self.min_puntos_sell,
            "stop_loss_pct":     self.stop_loss_pct,
            "warmup":            self.warmup,
            # Campos requeridos por Graficador.py
            "rsi_length":        self.rsi_period,
            "N":                 self.window,
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "guardia_compra":    True,
            "guardia_venta":     True,
        }
