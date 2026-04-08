"""
strategies/divergence_field_strategy.py — DivergenceFieldStrategy
══════════════════════════════════════════════════════════════════
Estrategia basada en Teoría de la Información aplicada a series de precios.

Componentes
───────────
  1. Transfer Entropy TE(taker_ratio → price_slope)
       Mide cuánta información "fluye" del flujo comprador hacia el precio.
       Alto TE en un turning point = volumen está "causando" el movimiento.
       Estimadores: binning (rápido), KDE (suave), k-NN Kraskov (preciso).

  2. Conditional Mutual Information CMI(RSI; vol_accel | price_vs_MA)
       Mide si RSI y aceleración del volumen se vuelven más acoplados
       condicionalmente al régimen del precio respecto a la MA20.
       Regímenes: binario (bajo/alto MA) o ternario (bajo/medio/alto).

  3. Divergence Field vectorial (Δprice, Δvol, Δtaker)
       ANALOGICAL: suma de aceleraciones (∂²/∂t² por componente)
         · price_div < 0 + vol_div > 0 → patrón BOTTOM
         · price_div > 0 + vol_div < 0 → patrón TOP
       JACOBIAN: eigenvalores de Cov([δp, δvol, δtaker])
         · eigenvector dominante con precio < 0, vol+taker > 0 → BOTTOM

  4. Sink condition: vol_last_k / vol_avg_ventana
       Confirmación de actividad reciente alta.
       Puede ser filtro AND (bloquea señal si no se cumple) o componente del score.

Todos los hiperparámetros son configurables vía DFConfig para optimización grid.

Compatibilidad: drop-in reemplazo de cualquier BaseStrategy del sistema.
"""

from __future__ import annotations

import numpy as np
from collections import Counter, deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple

from actors.price_feed        import Candle
from actors.wallet            import Wallet
from strategies.base_strategy import BaseStrategy, Signal, SignalSide, HOLD
from support.logger           import get_logger

log = get_logger("divergence_field")


# ══════════════════════════════════════════════════════════════════════════════
# ENUMS — opciones seleccionables para el grid
# ══════════════════════════════════════════════════════════════════════════════

class TEEstimator(str, Enum):
    BINNING = "binning"  # Binning equipercentil — rápido, ventana ≥ 10
    KDE     = "kde"      # Kernel Density Estimation — ventana ≥ 15
    KNN     = "knn"      # Kraskov k-NN (Frenzel-Pompe CMI) — ventana ≥ 20

class WindowMode(str, Enum):
    FIXED    = "fixed"     # ventana = window_size en todo momento
    ADAPTIVE = "adaptive"  # ventana escala inversamente con vol relativa

class FieldDefinition(str, Enum):
    ANALOGICAL = "analogical"  # aceleraciones → divergencia temporal
    JACOBIAN   = "jacobian"    # eigenvalores de Cov([δp, δvol, δtaker])

class CMIRegimes(int, Enum):
    BINARY  = 2   # price_vs_MA: bajo / alto
    TERNARY = 3   # price_vs_MA: bajo / medio / alto

class ThresholdMode(str, Enum):
    ADAPTIVE_PERCENTILE = "adaptive_percentile"  # percentil rolling vs historia
    FIXED               = "fixed"                # tanh(valor_crudo) normalizado

class SinkMode(str, Enum):
    FILTER_AND      = "filter_and"      # AND duro: sink < threshold → no señal
    SCORE_COMPONENT = "score_component" # sumado ponderado al score final


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — dataclass con todos los hiperparámetros
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class DFConfig:
    """
    Configuración completa de DivergenceFieldStrategy.
    Todos los campos tienen defaults funcionales.
    Usada directamente por el grid optimizer.
    """
    # ── Estimador y ventana ───────────────────────────────────────────────────
    te_estimator:        TEEstimator      = TEEstimator.BINNING
    window_mode:         WindowMode       = WindowMode.FIXED
    window_size:         int              = 20     # ventana base (10–30)
    window_min:          int              = 10     # clamp inferior para ADAPTIVE
    window_max:          int              = 40     # clamp superior para ADAPTIVE

    # ── Campo vectorial ───────────────────────────────────────────────────────
    field_def:           FieldDefinition  = FieldDefinition.ANALOGICAL
    cmi_regimes:         CMIRegimes       = CMIRegimes.BINARY

    # ── Normalización y umbrales ──────────────────────────────────────────────
    threshold_mode:      ThresholdMode    = ThresholdMode.ADAPTIVE_PERCENTILE
    te_threshold:        float            = 0.70   # TE normalizada para señal
    cmi_threshold:       float            = 0.60   # CMI normalizada para señal
    field_threshold:     float            = 0.50   # |div normalizada| para señal

    # ── Condición sumidero ────────────────────────────────────────────────────
    sink_mode:           SinkMode         = SinkMode.SCORE_COMPONENT
    sink_threshold:      float            = 1.20   # vol_last_k / vol_avg
    sink_window:         int              = 5      # velas para el promedio reciente

    # ── Score y señal ─────────────────────────────────────────────────────────
    w_te:                float            = 0.40   # peso TE en score final
    w_cmi:               float            = 0.30   # peso CMI
    w_field:             float            = 0.20   # peso |divergencia del campo|
    w_sink:              float            = 0.10   # peso sink (si SCORE_COMPONENT)
    score_threshold_bot: float            = 0.55   # score mínimo → BUY
    score_threshold_top: float            = 0.55   # score mínimo → SELL

    # ── Cooldown ─────────────────────────────────────────────────────────────
    cooldown:            int              = 0      # velas entre señales

    # ── Hiperparámetros de estimadores ────────────────────────────────────────
    k_bins:              int              = 4      # bins para binning TE/CMI
    k_nn:                int              = 3      # vecinos para KNN
    n_norm:              int              = 200    # historia rolling (normalización)

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
# MATH — utilidades de discretización
# ══════════════════════════════════════════════════════════════════════════════

def _digitize_pct(arr: np.ndarray, k_bins: int) -> np.ndarray:
    """
    Discretiza arr en k_bins bins equipercentil.
    Retorna enteros en [0, k_bins-1]. Maneja duplicados en los bordes.
    """
    if len(arr) < 2:
        return np.zeros(len(arr), dtype=np.int32)
    quantiles = np.linspace(0, 100, k_bins + 1)[1:-1]
    edges = np.unique(np.percentile(arr, quantiles))
    return np.searchsorted(edges, arr).astype(np.int32)


# ══════════════════════════════════════════════════════════════════════════════
# MATH — Transfer Entropy (3 estimadores)
# ══════════════════════════════════════════════════════════════════════════════

def te_binning(x_arr: np.ndarray, y_arr: np.ndarray,
               k_bins: int = 4, lag: int = 1) -> float:
    """
    TE(X → Y) via binning equipercentil.

    TE = Σ p(y,y₋₁,x₋₁) · log₂[ p(y|y₋₁,x₋₁) / p(y|y₋₁) ]
       = H(Y_t|Y_{t-1}) − H(Y_t|Y_{t-1}, X_{t-1})

    Complejidad: O(n). Funcional con ventana ≥ 10.
    """
    n = min(len(x_arr), len(y_arr))
    if n < lag + k_bins + 1:
        return 0.0

    xd = _digitize_pct(x_arr[:n], k_bins)
    yd = _digitize_pct(y_arr[:n], k_bins)

    yt  = yd[lag:]
    yt1 = yd[:n - lag]
    xt1 = xd[:n - lag]
    m   = len(yt)

    c3   = Counter(zip(yt.tolist(), yt1.tolist(), xt1.tolist()))
    c_yy = Counter(zip(yt.tolist(),  yt1.tolist()))
    c_yx = Counter(zip(yt1.tolist(), xt1.tolist()))
    c_y1 = Counter(yt1.tolist())

    total = float(m)
    te    = 0.0

    for (y, y1, x1), cnt in c3.items():
        p_3    = cnt / total
        p_y1x1 = c_yx.get((y1, x1), 0) / total
        p_yy1  = c_yy.get((y, y1),  0) / total
        p_y1   = c_y1.get(y1,        0) / total
        if p_y1x1 > 0 and p_yy1 > 0 and p_y1 > 0:
            ratio = (p_3 * p_y1) / (p_y1x1 * p_yy1)
            if ratio > 1e-12:
                te += p_3 * np.log2(ratio)

    return max(0.0, te)


def te_kde(x_arr: np.ndarray, y_arr: np.ndarray, lag: int = 1) -> float:
    """
    TE(X → Y) via Kernel Density Estimation (Gaussian, bw=Scott).
    Estima densidades conjuntas y marginales con scipy.gaussian_kde.
    Complejidad: O(n²). Requiere ventana ≥ 15.
    Fallback automático a binning si scipy no está disponible.
    """
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

    def _norm(a: np.ndarray) -> np.ndarray:
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

        mask  = (p3 > 1e-12) & (pyy > 1e-12) & (pyx > 1e-12) & (py1 > 1e-12)
        if mask.sum() < 4:
            return 0.0

        ratio = (p3[mask] * py1[mask]) / (pyx[mask] * pyy[mask])
        return max(0.0, float(np.mean(np.log2(np.maximum(ratio, 1e-12)))))
    except Exception:
        return te_binning(x_arr, y_arr, lag=lag)


def te_knn(x_arr: np.ndarray, y_arr: np.ndarray,
           k: int = 3, lag: int = 1) -> float:
    """
    TE(X → Y) via estimador Kraskov/Frenzel-Pompe (CMI k-NN).
    TE = CMI(Y_t; X_{t-1} | Y_{t-1})
       = ψ(k) − ⟨ψ(n_yy1+1)⟩ − ⟨ψ(n_y1x1+1)⟩ + ⟨ψ(n_y1+1)⟩

    Complejidad: O(n² × d) vectorizado. Requiere ventana ≥ 20 y sklearn.
    Fallback automático a binning.
    """
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

    def _norm(a: np.ndarray) -> np.ndarray:
        s = a.std()
        return (a - a.mean()) / s if s > 1e-10 else np.zeros_like(a)

    yn  = _norm(yt).reshape(-1, 1)
    y1n = _norm(yt1).reshape(-1, 1)
    x1n = _norm(xt1).reshape(-1, 1)

    s3d = np.hstack([yn, y1n, x1n])
    syy = np.hstack([yn, y1n])
    syx = np.hstack([y1n, x1n])

    try:
        nn3  = NearestNeighbors(n_neighbors=k + 1, metric='chebyshev').fit(s3d)
        dist, _ = nn3.kneighbors(s3d)
        eps  = dist[:, k] + 1e-10

        # Conteo vectorizado en Chebyshev — O(n² × d)
        def _count_cheby(space: np.ndarray, eps_arr: np.ndarray) -> np.ndarray:
            # diffs[i,j] = Chebyshev(space[i], space[j])
            d = np.abs(space[:, np.newaxis, :] - space[np.newaxis, :, :]).max(axis=2)
            return np.maximum((d < eps_arr[:, np.newaxis]).sum(axis=1) - 1, 0)

        n_yy = _count_cheby(syy,       eps)
        n_yx = _count_cheby(syx,       eps)
        n_y1 = _count_cheby(y1n,       eps.reshape(-1, 1))  # 1D

        te = (float(digamma(k))
              - float(np.mean(digamma(np.maximum(n_yy + 1, 1))))
              - float(np.mean(digamma(np.maximum(n_yx + 1, 1))))
              + float(np.mean(digamma(np.maximum(n_y1 + 1, 1)))))

        return max(0.0, te / np.log(2))  # nats → bits
    except Exception:
        return te_binning(x_arr, y_arr, lag=lag)


# ══════════════════════════════════════════════════════════════════════════════
# MATH — Conditional Mutual Information
# ══════════════════════════════════════════════════════════════════════════════

def cmi_binning(a_arr: np.ndarray, b_arr: np.ndarray,
                c_arr: np.ndarray,
                n_regimes: int = 2, k_bins: int = 4) -> float:
    """
    CMI(A; B | C) via binning.

    C (price_vs_MA) se discretiza en n_regimes regímenes equipercentil.
    A (RSI) y B (vol_accel) se discretizan en k_bins bins.

    CMI = Σ_c p(c) · I(A; B | C=c)
        = Σ_c p(c) · [H(A|c) + H(B|c) − H(A,B|c)]

    Complejidad: O(n·k²). Estable con ventana ≥ 10.
    """
    n = min(len(a_arr), len(b_arr), len(c_arr))
    if n < k_bins + 2:
        return 0.0

    ad = _digitize_pct(a_arr[:n], k_bins)
    bd = _digitize_pct(b_arr[:n], k_bins)

    c_edges = np.unique(np.percentile(c_arr[:n],
                        np.linspace(0, 100, n_regimes + 1)[1:-1]))
    cd = np.searchsorted(c_edges, c_arr[:n]).astype(np.int32)

    total = float(n)
    cmi   = 0.0

    for regime in range(n_regimes):
        idx = np.where(cd == regime)[0]
        n_c = len(idx)
        if n_c < 4:
            continue
        p_c = n_c / total
        ac, bc = ad[idx], bd[idx]

        c_ab = Counter(zip(ac.tolist(), bc.tolist()))
        c_a  = Counter(ac.tolist())
        c_b  = Counter(bc.tolist())
        n_c_f = float(n_c)

        mi_c = 0.0
        for (ai, bi), cnt in c_ab.items():
            p_ab = cnt / n_c_f
            p_a  = c_a.get(ai, 0) / n_c_f
            p_b  = c_b.get(bi, 0) / n_c_f
            if p_a > 0 and p_b > 0:
                mi_c += p_ab * np.log2(p_ab / (p_a * p_b + 1e-12) + 1e-12)

        cmi += p_c * max(0.0, mi_c)

    return max(0.0, cmi)


# ══════════════════════════════════════════════════════════════════════════════
# MATH — Divergence Field
# ══════════════════════════════════════════════════════════════════════════════

def _build_field_vectors(window: List[Candle]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construye las diferencias primeras normalizadas del campo vectorial:
      δprice  = Δclose  / σ(Δclose)
      δvol    = Δvol    / σ(Δvol)
      δtaker  = Δtaker_ratio / σ(Δtaker_ratio)

    Fallback para taker_ratio: 0.5 si el campo no está disponible.
    Retorna tres arrays de longitud (len(window) − 1).
    """
    closes = np.array([c.close  for c in window], dtype=np.float64)
    vols   = np.array([c.volume for c in window], dtype=np.float64)
    takers = np.array([
        (c.taker_buy_base_vol / c.volume)
        if (c.taker_buy_base_vol is not None and c.volume > 1e-10)
        else 0.5
        for c in window
    ], dtype=np.float64)

    def _snorm(arr: np.ndarray) -> np.ndarray:
        s = arr.std()
        return arr / s if s > 1e-10 else arr

    return _snorm(np.diff(closes)), _snorm(np.diff(vols)), _snorm(np.diff(takers))


def field_analogical(window: List[Candle]) -> Tuple[float, float, float]:
    """
    Campo vectorial analógico: suma de aceleraciones por componente.

    price_div     = mean(Δ²price)  — negativo si precio decelera hacia abajo
    vol_taker_div = mean(Δ²vol) + mean(Δ²taker)  — positivo si volumen acelera

    Patrón BOTTOM: price_div < 0  AND vol_taker_div > 0
    Patrón TOP:    price_div > 0  AND vol_taker_div < 0

    curl_proxy: correlación cruzada entre δvol y d(δprice) − correlación inversa
                mide "rotación" del campo precio-volumen.

    Retorna: (price_div, vol_taker_div, curl_proxy)
    """
    if len(window) < 5:
        return 0.0, 0.0, 0.0

    dp, dv, dt = _build_field_vectors(window)
    if len(dp) < 4:
        return 0.0, 0.0, 0.0

    ddp = np.diff(dp)
    ddv = np.diff(dv)
    ddt = np.diff(dt)

    price_div     = float(np.mean(ddp))
    vol_taker_div = float(np.mean(ddv) + np.mean(ddt))

    # Curl proxy: rotación precio-volumen
    try:
        if len(dp) >= 5:
            m = min(len(dp) - 1, len(dv) - 1)
            curl = float(
                np.corrcoef(dv[:m], np.diff(dp[:m + 1]))[0, 1]
                - np.corrcoef(dp[:m], np.diff(dv[:m + 1]))[0, 1]
            )
        else:
            curl = 0.0
    except Exception:
        curl = 0.0

    # Limpiar NaN/inf
    price_div     = 0.0 if not np.isfinite(price_div)     else price_div
    vol_taker_div = 0.0 if not np.isfinite(vol_taker_div) else vol_taker_div
    curl          = 0.0 if not np.isfinite(curl)          else curl

    return price_div, vol_taker_div, curl


def field_jacobian(window: List[Candle]) -> Tuple[float, float, float]:
    """
    Campo vectorial vía descomposición espectral de Cov([δprice, δvol, δtaker]).

    J = Cov([δp, δvol, δtaker])  →  matriz 3×3
    Descomposición:  J = V · diag(λ) · Vᵀ

    El eigenvector del eigenvalor dominante indica la dirección de mayor
    varianza del campo:
      · Componente precio < 0 → precio cayendo (BOTTOM pattern)
      · vol + taker > 0       → volumen subiendo

    price_component     = signo(dom_eigenvec[0]) × |dom_eigenvec[0]|
    vol_taker_component = dom_eigenvec[1] + dom_eigenvec[2]
    curl_proxy          = corr(δprice, δvol)  (proyección rotacional)

    Retorna: (price_component, vol_taker_component, curl_proxy)
    """
    if len(window) < 6:
        return 0.0, 0.0, 0.0

    dp, dv, dt = _build_field_vectors(window)
    m = min(len(dp), len(dv), len(dt))
    if m < 5:
        return 0.0, 0.0, 0.0

    field_mat = np.vstack([dp[:m], dv[:m], dt[:m]]).T  # (m, 3)

    try:
        cov = np.cov(field_mat.T)
        if not np.all(np.isfinite(cov)):
            return 0.0, 0.0, 0.0

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        dom_idx = int(np.argmax(np.abs(eigenvalues)))
        dom_vec = eigenvectors[:, dom_idx]

        price_component     = float(dom_vec[0])
        vol_taker_component = float(dom_vec[1] + dom_vec[2])
        curl_proxy          = float(np.corrcoef(dp[:m], dv[:m])[0, 1])

        price_component     = 0.0 if not np.isfinite(price_component)     else price_component
        vol_taker_component = 0.0 if not np.isfinite(vol_taker_component) else vol_taker_component
        curl_proxy          = 0.0 if not np.isfinite(curl_proxy)          else curl_proxy

        return price_component, vol_taker_component, curl_proxy

    except (np.linalg.LinAlgError, ValueError):
        return 0.0, 0.0, 0.0


# ══════════════════════════════════════════════════════════════════════════════
# MATH — RSI helper
# ══════════════════════════════════════════════════════════════════════════════

def _rsi(closes: np.ndarray, period: int = 14) -> float:
    """RSI estándar [0, 100]. Retorna 50 si no hay suficientes datos."""
    if len(closes) < period + 1:
        return 50.0
    deltas = np.diff(closes[-(period + 1):])
    avg_gain = np.where(deltas > 0, deltas, 0.0).mean()
    avg_loss = np.where(deltas < 0, -deltas, 0.0).mean()
    if avg_loss < 1e-10:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ══════════════════════════════════════════════════════════════════════════════
# ESTRATEGIA
# ══════════════════════════════════════════════════════════════════════════════

class DivergenceFieldStrategy(BaseStrategy):
    """
    Estrategia completa basada en Transfer Entropy + CMI + Divergence Field.

    El score BOT y score TOP se calculan como combinación lineal ponderada de:
      te_n    (TE normalizada)     × w_te
      cmi_n   (CMI normalizada)    × w_cmi
      field_n (|divergencia| norm) × w_field   [solo si campo es consistente]
      sink_n  (vol reciente/avg)   × w_sink    [solo en SCORE_COMPONENT]

    Señal: SELL si score_top ≥ score_threshold_top  (prioridad)
           BUY  si score_bot ≥ score_threshold_bot
    """

    def __init__(self, config: DFConfig = None) -> None:
        super().__init__(name="DivergenceField-v1")
        self.cfg = config or DFConfig()
        self.cfg.validate()

        # Buffer máximo (window_max + margen para modo adaptativo)
        self._buf: Deque[Candle] = deque(maxlen=self.cfg.window_max + 10)

        # Historias rolling para normalización ADAPTIVE_PERCENTILE
        self._h_te:    List[float] = []
        self._h_cmi:   List[float] = []
        self._h_field: List[float] = []
        self._h_sink:  List[float] = []

        # Cooldown
        _NEG = -(10 ** 9)
        self._last_bot_idx: int = _NEG
        self._last_top_idx: int = _NEG

        # ── Atributos públicos (para enriquecer el trade_log) ─────────────────
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
        current_idx = self._candles_seen  # ya incrementado por _tick()

        win = self._effective_window()
        if len(self._buf) < win:
            self._reset_last()
            return HOLD

        window: List[Candle] = list(self._buf)[-win:]

        # ── Calcular features crudas ──────────────────────────────────────────
        te_raw                      = self._compute_te(window)
        cmi_raw                     = self._compute_cmi(window)
        price_div, vol_div, curl    = self._compute_field(window)
        sink_raw                    = self._compute_sink(window)

        # ── Actualizar historias ──────────────────────────────────────────────
        self._push_history(te_raw, cmi_raw, abs(price_div), sink_raw)

        # ── Normalizar a [0, 1] ───────────────────────────────────────────────
        te_n    = self._normalize(te_raw,         self._h_te)
        cmi_n   = self._normalize(cmi_raw,        self._h_cmi)
        field_n = self._normalize(abs(price_div), self._h_field)
        sink_n  = self._normalize(sink_raw,       self._h_sink)

        # ── Detectar patrón del campo ─────────────────────────────────────────
        # BOTTOM: precio diverge negativamente + volumen/taker positivamente
        # TOP:    precio diverge positivamente + volumen/taker negativamente
        is_bot_pattern = (price_div < 0.0) and (vol_div > 0.0)
        is_top_pattern = (price_div > 0.0) and (vol_div < 0.0)

        # ── Exponer atributos públicos ────────────────────────────────────────
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

        # ── Cooldown ──────────────────────────────────────────────────────────
        cd_ok_bot = (self.cfg.cooldown == 0 or
                     (current_idx - self._last_bot_idx) >= self.cfg.cooldown)
        cd_ok_top = (self.cfg.cooldown == 0 or
                     (current_idx - self._last_top_idx) >= self.cfg.cooldown)

        # ── Calcular scores ───────────────────────────────────────────────────
        score_bot, score_top = self._compute_scores(
            te_n, cmi_n, field_n, sink_n, sink_raw,
            is_bot_pattern, is_top_pattern
        )
        self.last_score_bot = score_bot
        self.last_score_top = score_top

        # ── Señal (SELL prioridad sobre BUY) ──────────────────────────────────
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
        """Ventana efectiva: fija o adaptativa por volatilidad relativa."""
        if self.cfg.window_mode == WindowMode.FIXED:
            return self.cfg.window_size

        # ADAPTIVE: ventana inversa a la volatilidad relativa
        n_buf = len(self._buf)
        if n_buf < 21:
            return self.cfg.window_size

        prices = np.array([c.close for c in self._buf], dtype=float)
        ret_s  = np.diff(np.log(prices[-20:]))
        ret_l  = np.diff(np.log(prices[-min(60, n_buf):]))
        vol_s  = ret_s.std() if len(ret_s) > 1 else 1.0
        vol_l  = ret_l.std() if len(ret_l) > 1 else 1.0
        vol_ratio = vol_s / (vol_l + 1e-10)
        # Alta vol → ventana pequeña (más responsiva)
        # Baja vol → ventana grande (más estable)
        win = int(self.cfg.window_size / max(0.4, min(2.5, vol_ratio)))
        return max(self.cfg.window_min, min(self.cfg.window_max, win))

    def _compute_te(self, window: List[Candle]) -> float:
        """TE(taker_ratio → price_slope) con el estimador configurado."""
        prices = np.array([c.close for c in window], dtype=float)
        takers = np.array([
            (c.taker_buy_base_vol / c.volume)
            if (c.taker_buy_base_vol is not None and c.volume > 1e-10)
            else 0.5
            for c in window
        ], dtype=float)

        price_slope  = np.diff(prices)
        taker_series = takers[:-1]

        if   self.cfg.te_estimator == TEEstimator.BINNING:
            return te_binning(taker_series, price_slope, k_bins=self.cfg.k_bins)
        elif self.cfg.te_estimator == TEEstimator.KDE:
            return te_kde(taker_series, price_slope)
        else:  # KNN
            return te_knn(taker_series, price_slope, k=self.cfg.k_nn)

    def _compute_cmi(self, window: List[Candle]) -> float:
        """CMI(RSI; vol_accel | price_vs_MA) con n_regimes configurado."""
        closes = np.array([c.close  for c in window], dtype=float)
        vols   = np.array([c.volume for c in window], dtype=float)
        n      = len(closes)

        # RSI en cada punto de la ventana
        rsi_vals = np.array([_rsi(closes[:i + 1]) for i in range(n)], dtype=float)

        # Aceleración del volumen (segunda diferencia)
        vol_accel = np.zeros(n, dtype=float)
        if n >= 3:
            vol_accel[2:] = np.diff(vols, 2)

        # price_vs_MA (MA20 rolling, o ventana completa si n < 20)
        ma_p = min(20, n)
        ma20 = np.array([
            closes[max(0, i - ma_p + 1):i + 1].mean()
            for i in range(n)
        ], dtype=float)
        price_vs_ma = (closes - ma20) / (ma20 + 1e-10)

        return cmi_binning(
            rsi_vals, vol_accel, price_vs_ma,
            n_regimes = int(self.cfg.cmi_regimes),
            k_bins    = self.cfg.k_bins,
        )

    def _compute_field(self, window: List[Candle]) -> Tuple[float, float, float]:
        """Calcula el campo vectorial según la definición configurada."""
        if self.cfg.field_def == FieldDefinition.ANALOGICAL:
            return field_analogical(window)
        else:
            return field_jacobian(window)

    def _compute_sink(self, window: List[Candle]) -> float:
        """
        vol_last_k / vol_avg_ventana.
        > 1.0 → actividad reciente superior al promedio de la ventana.
        """
        vols = np.array([c.volume for c in window], dtype=float)
        k    = self.cfg.sink_window
        if len(vols) < k + 1:
            return 1.0
        vol_avg = vols.mean()
        if vol_avg < 1e-10:
            return 1.0
        return float(vols[-k:].mean() / vol_avg)

    def _push_history(self, te: float, cmi: float,
                      field_abs: float, sink: float) -> None:
        n = self.cfg.n_norm
        for arr, val in [(self._h_te, te), (self._h_cmi, cmi),
                         (self._h_field, field_abs), (self._h_sink, sink)]:
            arr.append(val)
            if len(arr) > n:
                arr.pop(0)

    def _normalize(self, value: float, history: List[float]) -> float:
        """
        ADAPTIVE_PERCENTILE: percentil del valor dentro de la historia rolling.
        FIXED: tanh(valor) — mapea [0, ∞) → [0, 1) suavemente.
        """
        if self.cfg.threshold_mode == ThresholdMode.FIXED:
            return float(np.tanh(max(0.0, value)))

        # ADAPTIVE_PERCENTILE
        if len(history) < 5:
            return 0.5
        return float(np.mean(np.array(history) <= value))

    def _compute_scores(
        self,
        te_n: float, cmi_n: float, field_n: float, sink_n: float, sink_raw: float,
        is_bot: bool, is_top: bool,
    ) -> Tuple[float, float]:
        """
        Calcula score_bot y score_top.

        SCORE_COMPONENT: sink_n entra como componente ponderado.
        FILTER_AND:      si sink_raw < sink_threshold → scores = 0.
        """
        cfg = self.cfg

        # campo: solo contribuye si el patrón es consistente con la dirección
        field_bot = field_n if is_bot else 0.0
        field_top = field_n if is_top else 0.0

        if cfg.sink_mode == SinkMode.SCORE_COMPONENT:
            w_sum = cfg.w_te + cfg.w_cmi + cfg.w_field + cfg.w_sink
            if w_sum < 1e-10:
                return 0.0, 0.0
            s_bot = (cfg.w_te    * te_n
                     + cfg.w_cmi  * cmi_n
                     + cfg.w_field * field_bot
                     + cfg.w_sink  * sink_n) / w_sum
            s_top = (cfg.w_te    * te_n
                     + cfg.w_cmi  * cmi_n
                     + cfg.w_field * field_top
                     + cfg.w_sink  * sink_n) / w_sum
        else:  # FILTER_AND
            if sink_raw < cfg.sink_threshold:
                return 0.0, 0.0
            w_sum = cfg.w_te + cfg.w_cmi + cfg.w_field
            if w_sum < 1e-10:
                return 0.0, 0.0
            s_bot = (cfg.w_te    * te_n
                     + cfg.w_cmi  * cmi_n
                     + cfg.w_field * field_bot) / w_sum
            s_top = (cfg.w_te    * te_n
                     + cfg.w_cmi  * cmi_n
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
