"""
strategies/dual_reversal.py — Dual Reversal Strategy
══════════════════════════════════════════════════════
Arquitectura de dos capas calibrada sobre turning points ZigZag 5%
(99 BOTTOMs y 98 TOPs del período 2021-11-10 → 2022-11-22).

Arquitectura
────────────
La estrategia separa dos preguntas distintas:

  CAPA 1 — CONTEXTO
    ¿Estamos en zona de posible giro?
    Evalúa el estado acumulado del mercado durante las últimas WINDOW velas.
    Calcula un puntaje sobre condiciones de fondo. La capa se activa cuando
    el puntaje supera CTX_MIN_PTS y permanece activa hasta que se opera.

  CAPA 2 — DISPARADOR
    ¿El giro está ocurriendo AHORA, en esta vela?
    Evalúa cambios en las últimas LAST_N velas. Solo se evalúa cuando
    el contexto está activo. Calcula un segundo puntaje sobre señales
    de reversión inmediata.

La señal se emite cuando AMBAS capas están activas simultáneamente.

Fundamento empírico (del CSV de turning_features.json)
───────────────────────────────────────────────────────
Cobertura medida sobre los turning points reales:

  Combinación (contexto AND disparador)          Bot%    Top(FP)%
  below_MAs AND rsi_slope<-0.03 AND taker<0     51.5%    0.0%
  below_MAs AND rsi_slope<-0.03                 66.7%    2.0%
  below_MAs AND taker_slope<0                   68.7%    1.0%
  above_MAs AND rsi_slope>0.03 AND taker>0      26.5%    0.0%  ← espejo
  above_MAs AND taker_slope>0                   50.0%    0.0%  ← espejo

La arquitectura de dos capas reduce los falsos positivos a 0-2% mientras
mantiene cobertura del 50-70% de los turning points.

Simetría BUY / SELL
────────────────────
Todas las condiciones son exactamente espejadas:
  Contexto BUY  = espejo exacto de Contexto SELL
  Disparador BUY = espejo exacto de Disparador SELL
No hay stop-loss — la venta siempre la decide el modelo SELL.

Parámetros configurables en backtest/live (para grid search)
─────────────────────────────────────────────────────────────
  Indicadores:
    rsi_period    — período del RSI                  [10, 14, 20]
    ma_short      — período MA corta                 [15, 20, 25]
    ma_long       — período MA larga                 [40, 50, 60]
    window        — velas de historia para contexto  [16, 24, 32]
    last_n        — velas para el disparador         [3, 5, 7]

  Umbrales de contexto:
    ctx_rsi_buy   — RSI máximo para contexto BUY    [35, 40, 45]
    ctx_rsi_sell  — RSI mínimo para contexto SELL   [55, 60, 65]
    ctx_ma20_buy  — distancia mínima bajo MA20 (%)  [-1.5, -2.5, -3.5]
    ctx_ma20_sell — distancia mínima sobre MA20 (%) [1.5, 2.5, 3.5]
    ctx_min_pts   — puntaje mínimo de contexto      [2, 3, 4]

  Umbrales de disparador:
    trig_rsi_slope  — pendiente RSI para disparar   [0.02, 0.03, 0.05]
    trig_taker_slope— pendiente taker para disparar [True = solo signo]
    trig_wick       — mecha mínima de la vela actual[0.25, 0.28, 0.32]
    trig_min_pts    — puntaje mínimo de disparador  [2, 3]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional

from actors.price_feed        import Candle
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("dual_reversal")


# ═══════════════════════════════════════════════════════════════════
# FUNCIONES DE INDICADORES (puras, sin estado)
# ═══════════════════════════════════════════════════════════════════

def _rsi(closes: List[float], period: int) -> Optional[float]:
    n = len(closes)
    if n < period + 1:
        return None
    gains  = [max(closes[i] - closes[i-1], 0.0) for i in range(n-period, n)]
    losses = [max(closes[i-1] - closes[i], 0.0) for i in range(n-period, n)]
    ag, al = sum(gains)/period, sum(losses)/period
    if al == 0:
        return 100.0
    return 100.0 - 100.0 / (1.0 + ag / al)


def _sma(values: List[float], period: int) -> Optional[float]:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def _slope_norm(values: List[float]) -> float:
    """
    Pendiente lineal normalizada por la media.
    Replicación exacta de linear_slope_norm() de analyze_turning_points.py.
    """
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    if y_mean == 0:
        return 0.0
    num = sum((i - x_mean) * (values[i] - y_mean) for i in range(n))
    den = sum((i - x_mean) ** 2                    for i in range(n))
    return (num / den) / abs(y_mean) if den else 0.0


def _wick_lower(candle: Candle) -> float:
    """(close - low) / (high - low) — fracción de mecha inferior."""
    rng = candle.high - candle.low
    return (candle.close - candle.low) / rng if rng > 0 else 0.5


def _wick_upper(candle: Candle) -> float:
    """(high - close) / (high - low) — fracción de mecha superior."""
    rng = candle.high - candle.low
    return (candle.high - candle.close) / rng if rng > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════
# ESTADO CALCULADO POR VELA
# ═══════════════════════════════════════════════════════════════════

@dataclass(slots=True)
class _Computed:
    # Contexto (ventana = WINDOW velas)
    rsi:           Optional[float]   # RSI al cierre
    price_vs_ma20: Optional[float]   # (close - MA20) / MA20 * 100
    price_vs_ma50: Optional[float]   # (close - MA50) / MA50 * 100
    below_both:    bool              # close < MA20 y close < MA50
    above_both:    bool              # close > MA20 y close > MA50
    ma20_slope:    float             # pendiente MA20 sobre LAST_N velas
    pct_bearish:   float             # % velas bajistas en WINDOW velas

    # Disparador (ventana = LAST_N velas)
    rsi_slope:     float             # pendiente RSI sobre LAST_N
    taker_slope:   float             # pendiente taker_ratio sobre LAST_N
    taker_last_vs_avg: Optional[float]  # media taker LAST_N / media taker WINDOW
    vol_last_vs_avg:   Optional[float]  # media vol LAST_N / media vol WINDOW

    # Vela actual (para mecha)
    lower_wick:    float
    upper_wick:    float


# ═══════════════════════════════════════════════════════════════════
# ESTRATEGIA
# ═══════════════════════════════════════════════════════════════════

class DualReversalStrategy(BaseStrategy):
    """
    Reversión de mercado con arquitectura contexto + disparador.
    Simétrica: BUY y SELL usan la misma lógica en espejo.
    Sin stop-loss — la salida siempre la gestiona el modelo.
    """

    # ── Defaults calibrados empíricamente ────────────────────────
    DEFAULT_RSI_PERIOD     = 14
    DEFAULT_MA_SHORT       = 20
    DEFAULT_MA_LONG        = 50
    DEFAULT_WINDOW         = 24
    DEFAULT_LAST_N         = 5

    DEFAULT_CTX_RSI_BUY    = 40.0   # p75 de RSI en BOTTOMs = 38.4
    DEFAULT_CTX_RSI_SELL   = 60.0   # p25 de RSI en TOPs    = 54.4
    DEFAULT_CTX_MA20_BUY   = -2.0   # p50 de price_vs_ma20 en BOTTOMs = -2.47
    DEFAULT_CTX_MA20_SELL  =  2.0   # p50 de price_vs_ma20 en TOPs    = +1.85
    DEFAULT_CTX_MIN_PTS    = 3

    DEFAULT_TRIG_RSI_SLOPE =  0.03  # p25 de |rsi_slope| en BOTTOMs = 0.026
    DEFAULT_TRIG_WICK      =  0.28  # p50 de lower_wick en BOTTOMs  = 0.292
    DEFAULT_TRIG_MIN_PTS   = 2

    DEFAULT_WARMUP         = 60     # MA_LONG + RSI + LAST_N + margen

    def __init__(
        self,
        # Indicadores
        rsi_period:      int   = DEFAULT_RSI_PERIOD,
        ma_short:        int   = DEFAULT_MA_SHORT,
        ma_long:         int   = DEFAULT_MA_LONG,
        window:          int   = DEFAULT_WINDOW,
        last_n:          int   = DEFAULT_LAST_N,
        # Umbrales contexto
        ctx_rsi_buy:     float = DEFAULT_CTX_RSI_BUY,
        ctx_rsi_sell:    float = DEFAULT_CTX_RSI_SELL,
        ctx_ma20_buy:    float = DEFAULT_CTX_MA20_BUY,
        ctx_ma20_sell:   float = DEFAULT_CTX_MA20_SELL,
        ctx_min_pts:     int   = DEFAULT_CTX_MIN_PTS,
        # Umbrales disparador
        trig_rsi_slope:  float = DEFAULT_TRIG_RSI_SLOPE,
        trig_wick:       float = DEFAULT_TRIG_WICK,
        trig_min_pts:    int   = DEFAULT_TRIG_MIN_PTS,
        # Warmup
        warmup:          int   = DEFAULT_WARMUP,
    ) -> None:
        super().__init__(name="DualReversal-2Capas")

        self.rsi_period     = rsi_period
        self.ma_short       = ma_short
        self.ma_long        = ma_long
        self.window         = window
        self.last_n         = last_n
        self.ctx_rsi_buy    = ctx_rsi_buy
        self.ctx_rsi_sell   = ctx_rsi_sell
        self.ctx_ma20_buy   = ctx_ma20_buy
        self.ctx_ma20_sell  = ctx_ma20_sell
        self.ctx_min_pts    = ctx_min_pts
        self.trig_rsi_slope = trig_rsi_slope
        self.trig_wick      = trig_wick
        self.trig_min_pts   = trig_min_pts
        self.warmup         = warmup

        # Buffer: suficiente para todos los indicadores
        buf = max(rsi_period + last_n + 4,
                  ma_long    + last_n + 4,
                  window     + last_n + 4)

        self._closes  : Deque[float]           = deque(maxlen=buf)
        self._takers  : Deque[Optional[float]] = deque(maxlen=buf)
        self._volumes : Deque[float]           = deque(maxlen=buf)
        self._rsi_hist: Deque[Optional[float]] = deque(maxlen=last_n + 4)
        self._ma20_hist:Deque[Optional[float]] = deque(maxlen=last_n + 4)

        log.info(
            "DualReversalStrategy configurada",
            ctx_rsi=f"<{ctx_rsi_buy}/>{ ctx_rsi_sell}",
            ctx_ma20=f"<{ctx_ma20_buy}%/>{ctx_ma20_sell}%",
            ctx_pts=ctx_min_pts,
            trig_rsi_slope=trig_rsi_slope,
            trig_wick=trig_wick,
            trig_pts=trig_min_pts,
        )

    # ══════════════════════════════════════════════════════════════
    # INTERFAZ BaseStrategy
    # ══════════════════════════════════════════════════════════════

    def on_start(self, wallet: Wallet, **kwargs) -> None:
        self._closes.clear()
        self._takers.clear()
        self._volumes.clear()
        self._rsi_hist.clear()
        self._ma20_hist.clear()
        log.info("DualReversalStrategy iniciada")

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        # ── Actualizar buffers ───────────────────────────────────
        self._closes.append(candle.close)
        self._volumes.append(candle.volume)
        taker = (
            candle.taker_buy_base_vol / candle.volume
            if candle.taker_buy_base_vol is not None and candle.volume > 0
            else None
        )
        self._takers.append(taker)

        closes = list(self._closes)
        rsi_v  = _rsi(closes, self.rsi_period)
        ma20_v = _sma(closes, self.ma_short)
        self._rsi_hist.append(rsi_v)
        self._ma20_hist.append(ma20_v)

        if self.candles_seen < self.warmup:
            return HOLD

        c = self._compute(candle, closes)

        # ── SELL tiene prioridad (cierra antes de abrir) ─────────
        if wallet.positions_count > 0:
            sell = self._eval(c, candle, is_buy=False, wallet=wallet)
            if sell is not None:
                return sell

        buy = self._eval(c, candle, is_buy=True, wallet=wallet)
        if buy is not None:
            return buy

        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info("DualReversalStrategy detenida", velas=self.candles_seen)

    # ══════════════════════════════════════════════════════════════
    # CÓMPUTO DE INDICADORES
    # ══════════════════════════════════════════════════════════════

    def _compute(self, candle: Candle, closes: List[float]) -> _Computed:
        n = len(closes)

        # RSI actual
        rsi_v = list(self._rsi_hist)[-1] if self._rsi_hist else None

        # MAs
        ma20 = _sma(closes, self.ma_short)
        ma50 = _sma(closes, self.ma_long)
        c    = candle.close
        pma20 = (c - ma20) / ma20 * 100 if ma20 else None
        pma50 = (c - ma50) / ma50 * 100 if ma50 else None
        below = bool(ma20 and ma50 and c < ma20 and c < ma50)
        above = bool(ma20 and ma50 and c > ma20 and c > ma50)

        # MA20 slope sobre últimas LAST_N velas con valor
        ma20_vals = [v for v in self._ma20_hist if v is not None]
        ma20_slope = _slope_norm(ma20_vals[-self.last_n:]) if len(ma20_vals) >= 2 else 0.0

        # % velas bajistas en WINDOW velas
        win = closes[-self.window:]
        bearish = sum(1 for i in range(1, len(win)) if win[i] < win[i-1])
        pct_b = bearish / max(1, len(win) - 1) * 100

        # RSI slope sobre LAST_N
        rsi_vals = [v for v in self._rsi_hist if v is not None]
        rsi_slope = _slope_norm(rsi_vals[-self.last_n:]) if len(rsi_vals) >= 2 else 0.0

        # Taker slope y ratio vs media
        takers_all = [v for v in self._takers if v is not None]
        win_takers = takers_all[-self.window:]
        taker_mean_win = sum(win_takers) / len(win_takers) if win_takers else None
        last_n_tak = takers_all[-self.last_n:] if takers_all else []
        taker_slope = _slope_norm(last_n_tak) if len(last_n_tak) >= 2 else 0.0
        taker_last_vs = None
        if last_n_tak and taker_mean_win and taker_mean_win > 0:
            taker_last_vs = sum(last_n_tak) / len(last_n_tak) / taker_mean_win

        # Vol ratio
        vols_all = list(self._volumes)
        win_vols = vols_all[-self.window:]
        vol_mean_win = sum(win_vols) / len(win_vols) if win_vols else None
        last_n_vol = vols_all[-self.last_n:]
        vol_last_vs = None
        if last_n_vol and vol_mean_win and vol_mean_win > 0:
            vol_last_vs = sum(last_n_vol) / len(last_n_vol) / vol_mean_win

        return _Computed(
            rsi            = rsi_v,
            price_vs_ma20  = pma20,
            price_vs_ma50  = pma50,
            below_both     = below,
            above_both     = above,
            ma20_slope     = ma20_slope,
            pct_bearish    = pct_b,
            rsi_slope      = rsi_slope,
            taker_slope    = taker_slope,
            taker_last_vs_avg = taker_last_vs,
            vol_last_vs_avg   = vol_last_vs,
            lower_wick     = _wick_lower(candle),
            upper_wick     = _wick_upper(candle),
        )

    # ══════════════════════════════════════════════════════════════
    # EVALUACIÓN (BUY y SELL usan la misma función — espejo por is_buy)
    # ══════════════════════════════════════════════════════════════

    def _eval(
        self,
        c:      _Computed,
        candle: Candle,
        is_buy: bool,
        wallet: Wallet,
    ) -> Optional[Signal]:
        """
        Evalúa ambas capas para BUY o SELL.
        Todas las condiciones se invierten con is_buy=False.
        """

        # ── CAPA 1: CONTEXTO ─────────────────────────────────────
        ctx_pts = 0
        ctx_why = []

        if c.rsi is not None:
            # RSI en zona extrema
            # BUY: RSI < ctx_rsi_buy  (calibrado: p75 BOTTOMs = 38.4)
            # SELL: RSI > ctx_rsi_sell (calibrado: p25 TOPs    = 54.4)
            rsi_strong = self.ctx_rsi_buy - 5 if is_buy else self.ctx_rsi_sell + 5
            rsi_weak   = self.ctx_rsi_buy      if is_buy else self.ctx_rsi_sell
            if (is_buy and c.rsi < rsi_strong) or (not is_buy and c.rsi > rsi_strong):
                ctx_pts += 3
                ctx_why.append(f"RSI={c.rsi:.1f}({'<' if is_buy else '>'}){rsi_strong:.0f}[3]")
            elif (is_buy and c.rsi < rsi_weak) or (not is_buy and c.rsi > rsi_weak):
                ctx_pts += 2
                ctx_why.append(f"RSI={c.rsi:.1f}({'<' if is_buy else '>'}){rsi_weak:.0f}[2]")

        if c.price_vs_ma20 is not None:
            # Distancia a MA20
            # BUY: precio bajo MA20 — calibrado: p50 BOTTOMs = -2.47%
            # SELL: precio sobre MA20 — calibrado: p50 TOPs = +1.85%
            ma_strong = self.ctx_ma20_buy   if is_buy else self.ctx_ma20_sell
            ma_weak   = self.ctx_ma20_buy / 2 if is_buy else self.ctx_ma20_sell / 2
            if (is_buy  and c.price_vs_ma20 < ma_strong) or \
               (not is_buy and c.price_vs_ma20 > ma_strong):
                ctx_pts += 2
                ctx_why.append(f"Δma20={c.price_vs_ma20:.2f}%[2]")
            elif (is_buy  and c.price_vs_ma20 < ma_weak) or \
                 (not is_buy and c.price_vs_ma20 > ma_weak):
                ctx_pts += 1
                ctx_why.append(f"Δma20={c.price_vs_ma20:.2f}%[1]")

        # Posición vs ambas MAs (90% BOTTOMs / 77% TOPs con 8-2% FP)
        if (is_buy and c.below_both) or (not is_buy and c.above_both):
            ctx_pts += 2
            ctx_why.append("both_MAs[2]")

        # Pendiente MA20 (91% BOTTOMs / 75% TOPs con 26-9% FP)
        if (is_buy and c.ma20_slope < 0) or (not is_buy and c.ma20_slope > 0):
            ctx_pts += 1
            ctx_why.append(f"ma20_sl={c.ma20_slope:.4f}[1]")

        # % velas en dirección dominante (68% BOTTOMs / 64% TOPs con 3% FP)
        if (is_buy and c.pct_bearish > 55) or (not is_buy and c.pct_bearish < 45):
            ctx_pts += 1
            ctx_why.append(f"dir={c.pct_bearish:.0f}%bear[1]")

        # Contexto inactivo → no evaluar disparador
        if ctx_pts < self.ctx_min_pts:
            return None

        # ── CAPA 2: DISPARADOR ───────────────────────────────────
        trig_pts = 0
        trig_why = []

        # RSI slope: cayendo antes de bottom, subiendo antes de top
        # Calibrado: p25 BOTTOMs rsi_slope_last5 = -0.107, p50 = -0.057
        #            p50 TOPs    rsi_slope_last5  = +0.037
        if (is_buy  and c.rsi_slope < -self.trig_rsi_slope) or \
           (not is_buy and c.rsi_slope >  self.trig_rsi_slope):
            trig_pts += 3
            trig_why.append(f"rsi_sl={c.rsi_slope:.4f}[3]")
        elif (is_buy  and c.rsi_slope < 0) or \
             (not is_buy and c.rsi_slope > 0):
            trig_pts += 1
            trig_why.append(f"rsi_sl={c.rsi_slope:.4f}[1]")

        # Taker slope: presión compradora cayendo (capitulación), subiendo (climax)
        # Calibrado: 79%/69% BOTTOMs/TOPs con solo signo; 34%/28% con umbral más duro
        if (is_buy  and c.taker_slope < 0) or \
           (not is_buy and c.taker_slope > 0):
            trig_pts += 2
            trig_why.append(f"taker_sl={c.taker_slope:.5f}[2]")

        # Taker last_n vs avg: debilidad compradora antes de bottom / fuerza antes de top
        # Calibrado: p50 BOTTOMs taker_last5_vs_avg = 0.986, p50 TOPs = 1.009
        if c.taker_last_vs_avg is not None:
            if (is_buy  and c.taker_last_vs_avg < 0.985) or \
               (not is_buy and c.taker_last_vs_avg > 1.015):
                trig_pts += 1
                trig_why.append(f"taker_ratio={c.taker_last_vs_avg:.4f}[1]")

        # Mecha de la vela ACTUAL: rechazo del mínimo/máximo
        # Usar vela actual (no promedio del segmento) captura el momento exacto
        # Calibrado: lower_wick p50 BOTTOMs = 0.292 / upper_wick p50 TOPs = 0.299
        wick = c.lower_wick if is_buy else c.upper_wick
        if wick > self.trig_wick:
            trig_pts += 2
            wlabel = "low_wick" if is_buy else "up_wick"
            trig_why.append(f"{wlabel}={wick:.2f}[2]")

        # Volumen elevado en las últimas N velas (capitulación / climax)
        # Calibrado: vol_last5_vs_avg p50 BOTTOMs = 1.20, p50 TOPs = 1.11
        if c.vol_last_vs_avg is not None and c.vol_last_vs_avg > 1.1:
            trig_pts += 1
            trig_why.append(f"vol={c.vol_last_vs_avg:.2f}[1]")

        if trig_pts < self.trig_min_pts:
            return None

        # ── Emitir señal ─────────────────────────────────────────
        side   = SignalSide.BUY if is_buy else SignalSide.SELL
        total  = ctx_pts + trig_pts
        reason = (f"{'BUY' if is_buy else 'SELL'} "
                  f"ctx={ctx_pts}/{self.ctx_min_pts}[{','.join(ctx_why)}] "
                  f"trig={trig_pts}/{self.trig_min_pts}[{','.join(trig_why)}]")
        score  = round(min(1.0, total / 14.0), 3)

        log.debug(f"{'BUY' if is_buy else 'SELL'} signal",
                  ctx=ctx_pts, trig=trig_pts, score=score)
        return Signal(side=side, price=candle.close, reason=reason, score=score)

    # ══════════════════════════════════════════════════════════════
    # METADATA
    # ══════════════════════════════════════════════════════════════

    def describe(self) -> dict:
        return {
            "estrategia":      self.name,
            "rsi_period":      self.rsi_period,
            "ma_short":        self.ma_short,
            "ma_long":         self.ma_long,
            "window":          self.window,
            "last_n":          self.last_n,
            "ctx_rsi_buy":     self.ctx_rsi_buy,
            "ctx_rsi_sell":    self.ctx_rsi_sell,
            "ctx_ma20_buy":    self.ctx_ma20_buy,
            "ctx_ma20_sell":   self.ctx_ma20_sell,
            "ctx_min_pts":     self.ctx_min_pts,
            "trig_rsi_slope":  self.trig_rsi_slope,
            "trig_wick":       self.trig_wick,
            "trig_min_pts":    self.trig_min_pts,
            # Campos esperados por Graficador.py
            "rsi_length":        self.rsi_period,
            "N":                 self.window,
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "guardia_compra":    True,
            "guardia_venta":     True,
        }
