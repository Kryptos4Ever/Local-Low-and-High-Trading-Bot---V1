"""
strategies/divergence_field_strategy.py — DivergenceFieldStrategy
══════════════════════════════════════════════════════════════════
Estrategia basada en Teoría de la Información aplicada a series de precios.

Componentes
───────────
  1. Transfer Entropy TE(taker_ratio → price_slope)
  2. Conditional Mutual Information CMI(RSI; vol_accel | price_vs_MA)
  3. Divergence Field vectorial (Δprice, Δvol, Δtaker)
  4. Sink condition: vol_last_k / vol_avg_ventana

OPTIMIZACIONES (sin cambio de resultados)
──────────────────────────────────────────
  · cmi_binning: Counter(zip(...)) → np.bincount  (~3× más rápido)
  · _push_history: list.pop(0) O(n) → deque(maxlen) O(1)
  · _normalize: compatible con deque y list
"""

from __future__ import annotations

import collections
import numpy as np
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

from actors.price_feed        import Candle
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("divergence_field")


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS
# ══════════════════════════════════════════════════════════════════════════════

class TEEstimator(str, Enum):
    BINNING = "binning"
    KDE     = "kde"
    KNN     = "knn"

class WindowMode(str, Enum):
    FIXED    = "fixed"
    ADAPTIVE = "adaptive"

class FieldDefinition(str, Enum):
    ANALOGICAL = "analogical"
    JACOBIAN   = "jacobian"

class CMIRegimes(int, Enum):
    BINARY  = 2
    TERNARY = 3

class ThresholdMode(str, Enum):
    ADAPTIVE_PERCENTILE = "adaptive_percentile"
    FIXED               = "fixed"

class SinkMode(str, Enum):
    FILTER_AND      = "filter_and"
    SCORE_COMPONENT = "score_component"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DFConfig:
    te_estimator:        TEEstimator      = TEEstimator.BINNING
    window_mode:         WindowMode       = WindowMode.FIXED
    window_size:         int              = 20
    window_min:          int              = 10
    window_max:          int              = 40
    field_def:           FieldDefinition  = FieldDefinition.ANALOGICAL
    cmi_regimes:         CMIRegimes       = CMIRegimes.BINARY
    threshold_mode:      ThresholdMode    = ThresholdMode.ADAPTIVE_PERCENTILE
    te_threshold:        float            = 0.70
    cmi_threshold:       float            = 0.60
    field_threshold:     float            = 0.50
    sink_mode:           SinkMode         = SinkMode.SCORE_COMPONENT
    sink_threshold:      float            = 1.20
    sink_window:         int              = 5
    w_te:                float            = 0.40
    w_cmi:               float            = 0.30
    w_field:             float            = 0.20
    w_sink:              float            = 0.10
    score_threshold_bot: float            = 0.55
    score_threshold_top: float            = 0.55
    cooldown:            int              = 0
    k_bins:              int              = 4
    k_nn:                int              = 3
    n_norm:              int              = 200

    def validate(self) -> None:
        assert self.window_size >= 8,    "window_size >= 8"
        assert self.window_min  >= 5,    "window_min >= 5"
        assert 0 < self.score_threshold_bot <= 1
        assert 0 < self.score_threshold_top <= 1
        assert self.te_estimator != TEEstimator.KNN or self.window_size >= 15, \
            "KNN requiere window_size >= 15"

    def to_dict(self) -> dict:
        return {
            "te_estimator":        self.te_estimator.value,
            "window_mode":         self.window_mode.value,
            "window_size":         self.window_size,
            "field_def":           self.field_def.value,
            "cmi_regimes":         int(self.cmi_regimes),
            "threshold_mode":      self.threshold_mode.value,
            "te_threshold":        self.te_threshold,
            "cmi_threshold":       self.cmi_threshold,
            "field_threshold":     self.field_threshold,
            "sink_mode":           self.sink_mode.value,
            "sink_threshold":      self.sink_threshold,
            "score_threshold_bot": self.score_threshold_bot,
            "score_threshold_top": self.score_threshold_top,
            "cooldown":            self.cooldown,
            "k_bins":              self.k_bins,
            "k_nn":                self.k_nn,
        }


# ══════════════════════════════════════════════════════════════════════════════
# MATH — discretización
# ══════════════════════════════════════════════════════════════════════════════

def _digitize_pct(arr: np.ndarray, k_bins: int) -> np.ndarray:
    if len(arr) < 2:
        return np.zeros(len(arr), dtype=np.int32)
    quantiles = np.linspace(0, 100, k_bins + 1)[1:-1]
    edges = np.unique(np.percentile(arr, quantiles))
    return np.searchsorted(edges, arr).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# MATH — Transfer Entropy
# ══════════════════════════════════════════════════════════════════════════════

def te_binning(x_arr: np.ndarray, y_arr: np.ndarray,
               k_bins: int = 4, lag: int = 1) -> float:
    """
    TE(X→Y) via binning equipercentil — versión vectorizada con np.bincount.
    Resultado idéntico al original con Counter; ~2–3× más rápido.
    """
    n = min(len(x_arr), len(y_arr))
    if n < lag + k_bins + 1:
        return 0.0

    xd = _digitize_pct(x_arr[:n], k_bins).astype(np.int32)
    yd = _digitize_pct(y_arr[:n], k_bins).astype(np.int32)

    yt  = yd[lag:]
    yt1 = yd[:n - lag]
    xt1 = xd[:n - lag]
    k   = k_bins
    m   = len(yt)

    c3   = np.bincount(yt * k*k + yt1 * k + xt1, minlength=k ** 3)
    c_yy = np.bincount(yt * k   + yt1,            minlength=k * k)
    c_yx = np.bincount(yt1* k   + xt1,            minlength=k * k)
    c_y1 = np.bincount(yt1,                        minlength=k)

    total = float(m)
    p3   = c3   / total
    p_yy = c_yy / total
    p_yx = c_yx / total
    p_y1 = c_y1 / total

    nz = np.nonzero(c3)[0]
    if nz.size == 0:
        return 0.0

    yt_v  = (nz // (k * k)).astype(np.int32)
    yt1_v = ((nz // k) % k).astype(np.int32)
    xt1_v = (nz % k).astype(np.int32)

    p3_v   = p3[nz]
    p_yy_v = p_yy[yt_v  * k + yt1_v]
    p_yx_v = p_yx[yt1_v * k + xt1_v]
    p_y1_v = p_y1[yt1_v]

    mask = (p_yy_v > 0) & (p_yx_v > 0) & (p_y1_v > 0)
    if not mask.any():
        return 0.0

    ratio = (p3_v[mask] * p_y1_v[mask]) / (p_yy_v[mask] * p_yx_v[mask])
    return max(0.0, float(np.sum(p3_v[mask] * np.log2(np.maximum(ratio, 1e-12)))))


def te_kde(x_arr: np.ndarray, y_arr: np.ndarray, lag: int = 1) -> float:
    try:
        from scipy.stats import gaussian_kde
    except ImportError:
        return te_binning(x_arr, y_arr, lag=lag)

    n = min(len(x_arr), len(y_arr))
    if n < lag + 8:
        return 0.0

    yt  = y_arr[lag:n]
    yt1 = y_arr[:n - lag]
    xt1 = x_arr[:n - lag]
    m   = min(len(yt), len(yt1))
    yt, yt1, xt1 = yt[:m], yt1[:m], xt1[:m]

    def _norm(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)

    yn, y1n, x1n = _norm(yt), _norm(yt1), _norm(xt1)

    try:
        kde_3  = gaussian_kde(np.vstack([yn, y1n, x1n]))
        kde_yy = gaussian_kde(np.vstack([yn, y1n]))
        kde_yx = gaussian_kde(np.vstack([y1n, x1n]))
        kde_y1 = gaussian_kde(y1n.reshape(1, -1))

        p3  = kde_3 (np.vstack([yn, y1n, x1n]))
        pyy = kde_yy(np.vstack([yn, y1n]))
        pyx = kde_yx(np.vstack([y1n, x1n]))
        py1 = kde_y1(y1n.reshape(1, -1))[0]

        mask = (p3 > 1e-12) & (pyy > 1e-12) & (pyx > 1e-12) & (py1 > 1e-12)
        if mask.sum() < 4:
            return 0.0

        ratio = (p3[mask] * py1[mask]) / (pyx[mask] * pyy[mask])
        return max(0.0, float(np.mean(np.log2(np.maximum(ratio, 1e-12)))))
    except Exception:
        return te_binning(x_arr, y_arr, lag=lag)


def te_knn(x_arr: np.ndarray, y_arr: np.ndarray,
           k: int = 3, lag: int = 1) -> float:
    try:
        from sklearn.neighbors import NearestNeighbors
        from scipy.special     import digamma
    except ImportError:
        return te_binning(x_arr, y_arr, lag=lag)

    n = min(len(x_arr), len(y_arr))
    if n < lag + k + 6:
        return 0.0

    yt  = y_arr[lag:n]
    yt1 = y_arr[:n - lag]
    xt1 = x_arr[:n - lag]
    m   = min(len(yt), len(yt1))
    yt, yt1, xt1 = yt[:m], yt1[:m], xt1[:m]

    def _norm(a):
        s = a.std()
        return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)

    yn  = _norm(yt).reshape(-1, 1)
    y1n = _norm(yt1).reshape(-1, 1)
    x1n = _norm(xt1).reshape(-1, 1)

    s3d = np.hstack([yn, y1n, x1n])
    syy = np.hstack([yn, y1n])
    syx = np.hstack([y1n, x1n])

    try:
        nn3 = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(s3d)
        dist, _ = nn3.kneighbors(s3d)
        eps = dist[:, k] + 1e-10

        def _count_cheby(space, eps_arr):
            d = np.abs(space[:, np.newaxis, :] - space[np.newaxis, :, :]).max(axis=2)
            return np.maximum((d < eps_arr[:, np.newaxis]).sum(axis=1) - 1, 0)

        n_yy = _count_cheby(syy, eps)
        n_yx = _count_cheby(syx, eps)
        n_y1 = _count_cheby(y1n, eps.reshape(-1, 1))

        te = (float(digamma(k))
              - float(np.mean(digamma(np.maximum(n_yy + 1, 1))))
              - float(np.mean(digamma(np.maximum(n_yx + 1, 1))))
              + float(np.mean(digamma(np.maximum(n_y1 + 1, 1)))))
        return max(0.0, te / np.log(2))
    except Exception:
        return te_binning(x_arr, y_arr, lag=lag)


# ══════════════════════════════════════════════════════════════════════════════
# MATH — Conditional Mutual Information (vectorizado)
# ══════════════════════════════════════════════════════════════════════════════

def cmi_binning(a_arr: np.ndarray, b_arr: np.ndarray,
                c_arr: np.ndarray,
                n_regimes: int = 2, k_bins: int = 4) -> float:
    """
    CMI(A; B | C) via binning — versión vectorizada con np.bincount.

    Idéntica en resultado a la versión original con Counter Python.
    ~3× más rápida por eliminación de tuplas Python y Counter overhead.
    """
    n = min(len(a_arr), len(b_arr), len(c_arr))
    if n < k_bins + 2:
        return 0.0

    ad = _digitize_pct(a_arr[:n], k_bins).astype(np.int32)
    bd = _digitize_pct(b_arr[:n], k_bins).astype(np.int32)

    c_edges = np.unique(np.percentile(c_arr[:n],
                        np.linspace(0, 100, n_regimes + 1)[1:-1]))
    cd = np.searchsorted(c_edges, c_arr[:n]).astype(np.int32)

    total = float(n)
    cmi   = 0.0
    k     = k_bins

    for regime in range(n_regimes):
        idx = np.where(cd == regime)[0]
        n_c = len(idx)
        if n_c < 4:
            continue
        p_c = n_c / total
        ac, bc = ad[idx], bd[idx]
        n_c_f  = float(n_c)

        # Conteos vectorizados
        c_ab = np.bincount(ac * k + bc, minlength=k * k) / n_c_f
        c_a  = np.bincount(ac,          minlength=k)     / n_c_f
        c_b  = np.bincount(bc,          minlength=k)     / n_c_f

        # Operaciones solo en pares no-cero
        nz_ab = np.nonzero(c_ab)[0]
        if nz_ab.size == 0:
            continue

        ai_v   = (nz_ab // k).astype(np.int32)
        bi_v   = (nz_ab %  k).astype(np.int32)
        p_ab_v = c_ab[nz_ab]
        p_a_v  = c_a[ai_v]
        p_b_v  = c_b[bi_v]

        mask = (p_a_v > 0) & (p_b_v > 0)
        if not mask.any():
            continue

        mi_c = float(np.sum(
            p_ab_v[mask] * np.log2(
                p_ab_v[mask] / (p_a_v[mask] * p_b_v[mask] + 1e-12) + 1e-12
            )
        ))
        cmi += p_c * max(0.0, mi_c)

    return max(0.0, cmi)


# ══════════════════════════════════════════════════════════════════════════════
# MATH — Divergence Field
# ══════════════════════════════════════════════════════════════════════════════

def _build_field_vectors(window: List[Candle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    closes = np.array([c.close  for c in window], dtype=np.float64)
    vols   = np.array([c.volume for c in window], dtype=np.float64)
    takers = np.array([
        (c.taker_buy_base_vol / c.volume)
        if (c.taker_buy_base_vol is not None and c.volume > 1e-10) else 0.5
        for c in window
    ], dtype=np.float64)

    def _snorm(arr):
        s = arr.std()
        return arr / s if s > 1e-10 else arr

    return _snorm(np.diff(closes)), _snorm(np.diff(vols)), _snorm(np.diff(takers))


def field_analogical(window: List[Candle]) -> Tuple[float, float, float]:
    if len(window) < 5:
        return 0.0, 0.0, 0.0
    dp, dv, dt = _build_field_vectors(window)
    if len(dp) < 4:
        return 0.0, 0.0, 0.0
    ddp = np.diff(dp); ddv = np.diff(dv); ddt = np.diff(dt)
    price_div     = float(np.mean(ddp))
    vol_taker_div = float(np.mean(ddv) + np.mean(ddt))
    try:
        if len(dp) >= 5:
            m    = min(len(dp) - 1, len(dv) - 1)
            curl = float(
                np.corrcoef(dv[:m], np.diff(dp[:m + 1]))[0, 1]
                - np.corrcoef(dp[:m], np.diff(dv[:m + 1]))[0, 1]
            )
        else:
            curl = 0.0
    except Exception:
        curl = 0.0
    price_div     = 0.0 if not np.isfinite(price_div)     else price_div
    vol_taker_div = 0.0 if not np.isfinite(vol_taker_div) else vol_taker_div
    curl          = 0.0 if not np.isfinite(curl)          else curl
    return price_div, vol_taker_div, curl


def field_jacobian(window: List[Candle]) -> Tuple[float, float, float]:
    if len(window) < 6:
        return 0.0, 0.0, 0.0
    dp, dv, dt = _build_field_vectors(window)
    m = min(len(dp), len(dv), len(dt))
    if m < 5:
        return 0.0, 0.0, 0.0
    field_mat = np.vstack([dp[:m], dv[:m], dt[:m]]).T
    try:
        cov = np.cov(field_mat.T)
        if not np.all(np.isfinite(cov)):
            return 0.0, 0.0, 0.0
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        dom_idx = int(np.argmax(np.abs(eigenvalues)))
        dom_vec = eigenvectors[:, dom_idx]
        pc   = float(dom_vec[0])
        vtc  = float(dom_vec[1] + dom_vec[2])
        curl = float(np.corrcoef(dp[:m], dv[:m])[0, 1])
        pc   = 0.0 if not np.isfinite(pc)   else pc
        vtc  = 0.0 if not np.isfinite(vtc)  else vtc
        curl = 0.0 if not np.isfinite(curl) else curl
        return pc, vtc, curl
    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MATH — RSI
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    if len(closes) < period + 1:
        return 50.0
    deltas   = np.diff(closes[-(period + 1):])
    avg_gain = np.where(deltas > 0, deltas, 0.0).mean()
    avg_loss = np.where(deltas < 0, -deltas, 0.0).mean()
    if avg_loss < 1e-10:
        return 100.0
    return 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA
# ══════════════════════════════════════════════════════════════════════════════

class DivergenceFieldStrategy(BaseStrategy):
    """
    Estrategia TE + CMI + Divergence Field.

    Cambios de rendimiento respecto a versión anterior (sin cambio de API):
      · _h_te/_h_cmi/_h_field/_h_sink: deque(maxlen) en lugar de list.
        Elimina list.pop(0) O(n) → O(1) por vela.
      · cmi_binning: vectorizado con np.bincount (~3× por call).
      · te_binning: vectorizado con np.bincount (~2–3× por call).
    """

    def __init__(self, config: DFConfig = None) -> None:
        super().__init__(name="DivergenceField-v1")
        self.cfg = config or DFConfig()
        self.cfg.validate()

        self._buf: Deque[Candle] = deque(maxlen=self.cfg.window_max + 10)

        # deque(maxlen=n_norm): append O(1), sin pop(0) O(n)
        n = self.cfg.n_norm
        self._h_te:    deque = deque(maxlen=n)
        self._h_cmi:   deque = deque(maxlen=n)
        self._h_field: deque = deque(maxlen=n)
        self._h_sink:  deque = deque(maxlen=n)

        _NEG = -(10 ** 9)
        self._last_bot_idx: int = _NEG
        self._last_top_idx: int = _NEG

        self.last_te:             Optional[float] = None
        self.last_cmi:            Optional[float] = None
        self.last_field_price:    Optional[float] = None
        self.last_field_vol:      Optional[float] = None
        self.last_field_curl:     Optional[float] = None
        self.last_sink:           Optional[float] = None
        self.last_te_norm:        float = 0.0
        self.last_cmi_norm:       float = 0.0
        self.last_field_norm:     float = 0.0
        self.last_sink_norm:      float = 0.0
        self.last_score_bot:      float = 0.0
        self.last_score_top:      float = 0.0
        self.last_is_bot_pattern: bool  = False
        self.last_is_top_pattern: bool  = False

        log.info("DivergenceFieldStrategy configurada", **self.cfg.to_dict())

    # ── Ciclo de vida ─────────────────────────────────────────────────────────

    def on_start(self, wallet: Wallet) -> None:
        self._buf.clear()
        self._h_te.clear(); self._h_cmi.clear()
        self._h_field.clear(); self._h_sink.clear()
        _NEG = -(10 ** 9)
        self._last_bot_idx = _NEG
        self._last_top_idx = _NEG
        self._reset_last()
        log.info("DivergenceFieldStrategy iniciada")

    def on_candle(self, candle: Candle, wallet: Wallet) -> Signal:
        self._buf.append(candle)
        current_idx = self._candles_seen

        win = self._effective_window()
        if len(self._buf) < win:
            self._reset_last()
            return HOLD

        window: List[Candle] = list(self._buf)[-win:]

        te_raw                   = self._compute_te(window)
        cmi_raw                  = self._compute_cmi(window)
        price_div, vol_div, curl = self._compute_field(window)
        sink_raw                 = self._compute_sink(window)

        self._push_history(te_raw, cmi_raw, abs(price_div), sink_raw)

        te_n    = self._normalize(te_raw,         self._h_te)
        cmi_n   = self._normalize(cmi_raw,        self._h_cmi)
        field_n = self._normalize(abs(price_div), self._h_field)
        sink_n  = self._normalize(sink_raw,       self._h_sink)

        is_bot_pattern = (price_div < 0.0) and (vol_div > 0.0)
        is_top_pattern = (price_div > 0.0) and (vol_div < 0.0)

        self.last_te             = te_raw
        self.last_cmi            = cmi_raw
        self.last_field_price    = price_div
        self.last_field_vol      = vol_div
        self.last_field_curl     = curl
        self.last_sink           = sink_raw
        self.last_te_norm        = te_n
        self.last_cmi_norm       = cmi_n
        self.last_field_norm     = field_n
        self.last_sink_norm      = sink_n
        self.last_is_bot_pattern = is_bot_pattern
        self.last_is_top_pattern = is_top_pattern

        cd_ok_bot = (self.cfg.cooldown == 0 or
                     (current_idx - self._last_bot_idx) >= self.cfg.cooldown)
        cd_ok_top = (self.cfg.cooldown == 0 or
                     (current_idx - self._last_top_idx) >= self.cfg.cooldown)

        score_bot, score_top = self._compute_scores(
            te_n, cmi_n, field_n, sink_n, sink_raw,
            is_bot_pattern, is_top_pattern
        )
        self.last_score_bot = score_bot
        self.last_score_top = score_top

        if score_top >= self.cfg.score_threshold_top and cd_ok_top:
            self._last_top_idx = current_idx
            return Signal(
                side   = SignalSide.SELL,
                price  = candle.close,
                reason = (f"divfield SELL "
                          f"te={te_n:.2f} cmi={cmi_n:.2f} "
                          f"field={field_n:.2f} sink={sink_n:.2f} "
                          f"score={score_top:.3f}"),
                score  = score_top,
            )

        if score_bot >= self.cfg.score_threshold_bot and cd_ok_bot:
            self._last_bot_idx = current_idx
            return Signal(
                side   = SignalSide.BUY,
                price  = candle.close,
                reason = (f"divfield BUY "
                          f"te={te_n:.2f} cmi={cmi_n:.2f} "
                          f"field={field_n:.2f} sink={sink_n:.2f} "
                          f"score={score_bot:.3f}"),
                score  = score_bot,
            )

        return HOLD

    def on_stop(self, wallet: Wallet) -> None:
        log.info("DivergenceFieldStrategy detenida",
                 velas_procesadas=self.candles_seen)

    # ── Helpers privados ──────────────────────────────────────────────────────

    def _effective_window(self) -> int:
        if self.cfg.window_mode == WindowMode.FIXED:
            return self.cfg.window_size
        n_buf = len(self._buf)
        if n_buf < 21:
            return self.cfg.window_size
        prices    = np.array([c.close for c in self._buf], dtype=float)
        ret_s     = np.diff(np.log(prices[-20:]))
        ret_l     = np.diff(np.log(prices[-min(60, n_buf):]))
        vol_s     = ret_s.std() if len(ret_s) > 1 else 1.0
        vol_l     = ret_l.std() if len(ret_l) > 1 else 1.0
        vol_ratio = vol_s / (vol_l + 1e-10)
        win       = int(self.cfg.window_size / max(0.4, min(2.5, vol_ratio)))
        return max(self.cfg.window_min, min(self.cfg.window_max, win))

    def _compute_te(self, window: List[Candle]) -> float:
        prices = np.array([c.close  for c in window], dtype=float)
        takers = np.array([
            (c.taker_buy_base_vol / c.volume)
            if (c.taker_buy_base_vol is not None and c.volume > 1e-10) else 0.5
            for c in window
        ], dtype=float)
        price_slope  = np.diff(prices)
        taker_series = takers[:-1]
        if   self.cfg.te_estimator == TEEstimator.BINNING:
            return te_binning(taker_series, price_slope, k_bins=self.cfg.k_bins)
        elif self.cfg.te_estimator == TEEstimator.KDE:
            return te_kde(taker_series, price_slope)
        else:
            return te_knn(taker_series, price_slope, k=self.cfg.k_nn)

    def _compute_cmi(self, window: List[Candle]) -> float:
        closes = np.array([c.close  for c in window], dtype=float)
        vols   = np.array([c.volume for c in window], dtype=float)
        n      = len(closes)

        rsi_vals = np.array([_rsi(closes[:i + 1]) for i in range(n)], dtype=float)

        vol_accel = np.zeros(n, dtype=float)
        if n >= 3:
            vol_accel[2:] = np.diff(vols, 2)

        ma_p = min(20, n)
        cum  = np.concatenate([[0.0], np.cumsum(closes)])
        ma20 = np.array([
            (cum[i + 1] - cum[max(0, i - ma_p + 1)]) / (i - max(0, i - ma_p + 1) + 1)
            for i in range(n)
        ], dtype=float)
        price_vs_ma = (closes - ma20) / (ma20 + 1e-10)

        return cmi_binning(
            rsi_vals, vol_accel, price_vs_ma,
            n_regimes=int(self.cfg.cmi_regimes),
            k_bins=self.cfg.k_bins,
        )

    def _compute_field(self, window: List[Candle]) -> Tuple[float, float, float]:
        if self.cfg.field_def == FieldDefinition.ANALOGICAL:
            return field_analogical(window)
        return field_jacobian(window)

    def _compute_sink(self, window: List[Candle]) -> float:
        vols    = np.array([c.volume for c in window], dtype=float)
        k       = self.cfg.sink_window
        if len(vols) < k + 1:
            return 1.0
        vol_avg = vols.mean()
        if vol_avg < 1e-10:
            return 1.0
        return float(vols[-k:].mean() / vol_avg)

    def _push_history(self, te: float, cmi: float,
                      field_abs: float, sink: float) -> None:
        # deque(maxlen=n_norm): append es O(1), descarte automático del extremo
        self._h_te.append(te)
        self._h_cmi.append(cmi)
        self._h_field.append(field_abs)
        self._h_sink.append(sink)

    def _normalize(self, value: float, history) -> float:
        if self.cfg.threshold_mode == ThresholdMode.FIXED:
            return float(np.tanh(max(0.0, value)))
        if len(history) < 5:
            return 0.5
        arr = np.asarray(history, dtype=np.float64)
        return float(np.mean(arr <= value))

    def _compute_scores(
        self,
        te_n: float, cmi_n: float, field_n: float, sink_n: float, sink_raw: float,
        is_bot: bool, is_top: bool,
    ) -> Tuple[float, float]:
        cfg = self.cfg
        field_bot = field_n if is_bot else 0.0
        field_top = field_n if is_top else 0.0

        if cfg.sink_mode == SinkMode.SCORE_COMPONENT:
            w_sum = cfg.w_te + cfg.w_cmi + cfg.w_field + cfg.w_sink
            if w_sum < 1e-10:
                return 0.0, 0.0
            s_bot = (cfg.w_te * te_n + cfg.w_cmi * cmi_n
                     + cfg.w_field * field_bot + cfg.w_sink * sink_n) / w_sum
            s_top = (cfg.w_te * te_n + cfg.w_cmi * cmi_n
                     + cfg.w_field * field_top + cfg.w_sink * sink_n) / w_sum
        else:
            if sink_raw < cfg.sink_threshold:
                return 0.0, 0.0
            w_sum = cfg.w_te + cfg.w_cmi + cfg.w_field
            if w_sum < 1e-10:
                return 0.0, 0.0
            s_bot = (cfg.w_te * te_n + cfg.w_cmi * cmi_n
                     + cfg.w_field * field_bot) / w_sum
            s_top = (cfg.w_te * te_n + cfg.w_cmi * cmi_n
                     + cfg.w_field * field_top) / w_sum

        return float(np.clip(s_bot, 0.0, 1.0)), float(np.clip(s_top, 0.0, 1.0))

    def _reset_last(self) -> None:
        self.last_te = self.last_cmi = self.last_field_price = None
        self.last_field_vol = self.last_field_curl = self.last_sink = None
        self.last_te_norm = self.last_cmi_norm = 0.0
        self.last_field_norm = self.last_sink_norm = 0.0
        self.last_score_bot = self.last_score_top = 0.0
        self.last_is_bot_pattern = self.last_is_top_pattern = False

    def describe(self) -> dict:
        d = self.cfg.to_dict()
        d.update({
            "estrategia":        self.name,
            "guardia_compra":    True,
            "guardia_venta":     True,
            "rsi_length":        "N/A",
            "ath_caida_maxima":  "N/A",
            "atl_subida_maxima": "N/A",
            "factor_caida":      "N/A",
            "factor_subida":     "N/A",
            "N":                 self.cfg.window_size,
        })
        return d
