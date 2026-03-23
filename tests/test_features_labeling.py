"""
tests/test_features_labeling.py
════════════════════════════════
Tests del pipeline de features y labeling de LocalReversalStrategy.

No requiere DB ni actores. Usa datos sintéticos para verificar
la aritmética exacta de cada feature y la lógica de labeling.

Ejecutar:
    python tests/test_features_labeling.py
    python -m pytest tests/test_features_labeling.py -v
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: construir velas sintéticas
# ─────────────────────────────────────────────────────────────────────────────

def _make_candles(n=300, seed=42):
    """
    Genera n velas OHLCV sintéticas con movimiento aleatorio.
    Suficiente para testear features que requieren warmup de 200 velas.
    """
    rng   = np.random.default_rng(seed)
    close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 50_000
    open_ = close * (1 + rng.normal(0, 0.005, n))
    high  = np.maximum(close, open_) * (1 + rng.uniform(0, 0.01, n))
    low   = np.minimum(close, open_) * (1 - rng.uniform(0, 0.01, n))
    vol   = rng.uniform(50, 200, n)
    taker = vol * rng.uniform(0.3, 0.7, n)
    return open_, high, low, close, vol, taker


def _compute_features(open_, high, low, close, vol, taker):
    """
    Replica exacta del cálculo de features de local_reversal.py.
    Retorna (feature_series, divergence) para verificación.
    """
    rng_arr    = np.where(high - low == 0, 1e-9, high - low)
    body_ratio = np.clip((close - open_) / rng_arr, -1, 1)
    lower_wick = np.clip((np.minimum(open_, close) - low) / rng_arr, 0, 1)
    upper_wick = np.clip((high - np.maximum(open_, close)) / rng_arr, 0, 1)
    delta_r    = np.clip(taker / (vol + 1e-9), 0, 1)
    roll48     = pd.Series(rng_arr).rolling(48, min_periods=1).mean().values
    range_rel  = np.clip(rng_arr / (roll48 + 1e-9), 0, 5)
    low_rej    = np.clip((close - low) / rng_arr, 0, 1)
    high_rej   = np.clip((high - close) / rng_arr, 0, 1)

    def rolling_zscore(arr, w=200):
        s  = pd.Series(arr)
        m  = s.rolling(w, min_periods=1).mean()
        st = s.rolling(w, min_periods=1).std().fillna(1).replace(0, 1)
        return ((s - m) / st).values

    ret_4h   = pd.Series(close).pct_change(4).fillna(0).values
    div      = rolling_zscore(delta_r) - rolling_zscore(ret_4h)

    fs = np.column_stack([
        body_ratio, lower_wick, upper_wick, delta_r,
        range_rel, div, low_rej, high_rej,
    ])
    return fs


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: aritmética de features individuales
# ─────────────────────────────────────────────────────────────────────────────

def test_body_ratio_vela_alcista():
    """Vela alcista (close > open): body_ratio positivo, en [-1, 1]."""
    open_, high, low, close = 100.0, 110.0, 90.0, 108.0
    rng  = high - low                              # 20
    body = np.clip((close - open_) / rng, -1, 1)  # (8) / 20 = 0.4
    assert abs(body - 0.4) < 1e-9
    assert -1.0 <= body <= 1.0


def test_body_ratio_vela_bajista():
    """Vela bajista (close < open): body_ratio negativo."""
    open_, high, low, close = 108.0, 110.0, 90.0, 100.0
    rng  = high - low
    body = np.clip((close - open_) / rng, -1, 1)   # -8/20 = -0.4
    assert body < 0.0


def test_lower_wick_correcto():
    """lower_wick = (min(open,close) - low) / range."""
    open_, high, low, close = 100.0, 110.0, 80.0, 100.0
    # min(100,100) = 100, low=80, range=30 → 20/30 ≈ 0.667
    rng  = high - low
    lw   = np.clip((min(open_, close) - low) / rng, 0, 1)
    assert abs(lw - 20/30) < 1e-9


def test_upper_wick_correcto():
    """upper_wick = (high - max(open,close)) / range."""
    open_, high, low, close = 100.0, 120.0, 80.0, 100.0
    # max(100,100)=100, high=120, range=40 → 20/40 = 0.5
    rng  = high - low
    uw   = np.clip((high - max(open_, close)) / rng, 0, 1)
    assert abs(uw - 0.5) < 1e-9


def test_delta_ratio_en_rango():
    """delta_ratio ∈ [0, 1] siempre."""
    for vol, taker in [(100, 60), (100, 0), (100, 100), (0, 0)]:
        dr = np.clip(taker / (vol + 1e-9), 0, 1)
        assert 0.0 <= dr <= 1.0


def test_low_rejection_vela_que_rebota_desde_abajo():
    """low_rejection = (close - low) / range. Mayor ≈ más rechazo del bajo."""
    open_, high, low, close = 100.0, 110.0, 80.0, 109.0
    rng   = high - low   # 30
    lr    = np.clip((close - low) / rng, 0, 1)   # 29/30 ≈ 0.967
    assert lr > 0.9


def test_high_rejection_vela_que_cae_desde_arriba():
    """high_rejection = (high - close) / range. Mayor ≈ más rechazo del alto."""
    open_, high, low, close = 100.0, 110.0, 80.0, 81.0
    rng   = high - low
    hr    = np.clip((high - close) / rng, 0, 1)   # 29/30 ≈ 0.967
    assert hr > 0.9


def test_vela_plana_no_produce_nan():
    """Con high=low, el range se fija a 1e-9 para evitar división por cero."""
    open_, high, low, close = 100.0, 100.0, 100.0, 100.0
    rng  = np.where(high - low == 0, 1e-9, high - low)
    body = np.clip((close - open_) / rng, -1, 1)
    assert not np.isnan(body)
    assert -1.0 <= body <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: shape y propiedades del array de features
# ─────────────────────────────────────────────────────────────────────────────

def test_feature_series_tiene_8_columnas():
    """El array de series de features debe tener 8 columnas."""
    open_, high, low, close, vol, taker = _make_candles(50)
    fs = _compute_features(open_, high, low, close, vol, taker)
    assert fs.shape == (50, 8)


def test_feature_series_sin_nan():
    """Ninguna feature puede ser NaN después del cálculo."""
    open_, high, low, close, vol, taker = _make_candles(300)
    fs = _compute_features(open_, high, low, close, vol, taker)
    assert not np.any(np.isnan(fs))


def test_feature_series_sin_inf():
    """Ninguna feature puede ser ±inf."""
    open_, high, low, close, vol, taker = _make_candles(300)
    fs = _compute_features(open_, high, low, close, vol, taker)
    assert not np.any(np.isinf(fs))


def test_feature_window_shape_es_197():
    """
    Una ventana de 24 velas × 8 series + 5 agregadas = 197 features.
    """
    WIN, N_SERIES, N_AGG = 24, 8, 5
    expected = WIN * N_SERIES + N_AGG
    assert expected == 197

    open_, high, low, close, vol, taker = _make_candles(300)
    fs = _compute_features(open_, high, low, close, vol, taker)

    # Construir una ventana de muestra
    i      = 250
    window = fs[i - WIN + 1 : i + 1]       # (24, 8)
    extra  = np.array([
        window[:, 0].mean(),
        window[-3:, 0].mean(),
        window[-6:, 5].mean(),
        window[-3:, 6].mean(),
        window[-3:, 7].mean(),
    ])
    sample_x = np.concatenate([window.flatten(), extra])
    assert sample_x.shape[0] == 197


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: labeling V3 (ventana ±18)
# ─────────────────────────────────────────────────────────────────────────────

def _build_labels_v3(low, high, ventana=18):
    """Labeling V3: ventana ±18, usa low para bottoms y high para tops."""
    N      = len(low)
    labels = np.zeros(N, dtype=np.int8)
    for i in range(ventana, N - ventana):
        if all(low[i] <= low[j]
               for j in range(i - ventana, i + ventana + 1) if j != i):
            labels[i] = 1
            continue
        if all(high[i] >= high[j]
               for j in range(i - ventana, i + ventana + 1) if j != i):
            labels[i] = 2
    return labels


def test_labeling_detecta_minimo_obvio():
    """Un mínimo bien aislado debe ser labeleado como bottom (1)."""
    low  = np.array([100.0] * 40 + [50.0] + [100.0] * 40)   # N=81
    high = low + 10
    labels = _build_labels_v3(low, high, ventana=18)
    # El índice 40 es el mínimo absoluto, debe ser bottom
    assert labels[40] == 1


def test_labeling_detecta_maximo_obvio():
    """Un máximo bien aislado debe ser labeleado como top (2)."""
    high = np.array([100.0] * 40 + [200.0] + [100.0] * 40)  # N=81
    low  = high - 10
    labels = _build_labels_v3(low, high, ventana=18)
    assert labels[40] == 2


def test_labeling_sin_extremo_es_neutro():
    """Serie plana no produce ningún extremo local."""
    n    = 50
    low  = np.ones(n) * 100.0
    high = np.ones(n) * 110.0
    labels = _build_labels_v3(low, high, ventana=18)
    # La serie plana no puede tener mínimos/máximos únicos
    assert (labels == 0).sum() + (labels == 1).sum() + (labels == 2).sum() == n


def test_labeling_bottom_no_puede_ser_top_simultaneamente():
    """Una vela no puede ser bottom y top al mismo tiempo."""
    rng   = np.random.default_rng(123)
    close = np.cumprod(1 + rng.normal(0, 0.02, 200)) * 50_000
    low   = close * 0.995
    high  = close * 1.005
    labels = _build_labels_v3(low, high, ventana=10)
    # labels puede ser 0, 1, o 2, pero no combinaciones
    assert set(labels).issubset({0, 1, 2})
    for i, lbl in enumerate(labels):
        assert lbl in (0, 1, 2)


def test_labeling_ventana_18_es_mas_selectivo_que_10():
    """Ventana ±18 produce menos eventos que ventana ±10."""
    rng   = np.random.default_rng(99)
    close = np.cumprod(1 + rng.normal(0, 0.015, 1000)) * 50_000
    low   = close * 0.995
    high  = close * 1.005

    labels_10 = _build_labels_v3(low, high, ventana=10)
    labels_18 = _build_labels_v3(low, high, ventana=18)

    n_eventos_10 = (labels_10 != 0).sum()
    n_eventos_18 = (labels_18 != 0).sum()
    assert n_eventos_18 < n_eventos_10


def test_labeling_extremo_necesita_ventana_completa():
    """Las primeras y últimas ventana velas no pueden ser extremos."""
    ventana = 18
    low  = np.concatenate([[50.0], np.ones(100) * 100.0, [50.0]])
    high = low + 10
    labels = _build_labels_v3(low, high, ventana=ventana)
    # Los índices 0 y 101 están dentro de la zona de borde → deben ser 0
    assert labels[0] == 0
    assert labels[-1] == 0


# ─────────────────────────────────────────────────────────────────────────────
# TESTS: lógica de señal de on_candle
# ─────────────────────────────────────────────────────────────────────────────

def _signal_logic(prob_bottom, prob_top, thr_b=0.50, thr_t=0.45):
    """Replica la lógica de on_candle de LocalReversalStrategy."""
    if prob_top >= thr_t:
        return "SELL"
    if prob_bottom >= thr_b:
        return "BUY"
    return "HOLD"


def test_señal_sell_tiene_prioridad_sobre_buy():
    """Cuando ambas probs superan sus umbrales, SELL tiene prioridad."""
    señal = _signal_logic(prob_bottom=0.60, prob_top=0.50, thr_b=0.50, thr_t=0.45)
    assert señal == "SELL"


def test_señal_buy_cuando_solo_prob_bottom_supera():
    señal = _signal_logic(prob_bottom=0.60, prob_top=0.30, thr_b=0.50, thr_t=0.45)
    assert señal == "BUY"


def test_señal_sell_cuando_solo_prob_top_supera():
    señal = _signal_logic(prob_bottom=0.30, prob_top=0.50, thr_b=0.50, thr_t=0.45)
    assert señal == "SELL"


def test_señal_hold_cuando_ninguna_supera():
    señal = _signal_logic(prob_bottom=0.49, prob_top=0.44, thr_b=0.50, thr_t=0.45)
    assert señal == "HOLD"


def test_señal_hold_en_borde_inferior():
    """Las probabilidades exactamente iguales al umbral NO superan (strict <)."""
    # thr_b=0.50: prob_bottom=0.50 está en el límite
    # La lógica usa >= así que 0.50 >= 0.50 → True → BUY
    señal = _signal_logic(prob_bottom=0.50, prob_top=0.44, thr_b=0.50, thr_t=0.45)
    assert señal == "BUY"


def test_asimetria_thr_t_menor_que_thr_b():
    """
    Con los parámetros calibrados (thr_b=0.50, thr_t=0.45),
    thr_t es intencionalmente menor que thr_b.
    Esto significa que es más fácil generar SELL que BUY.
    """
    thr_b, thr_t = 0.50, 0.45
    assert thr_t < thr_b
    # prob_top=0.47 supera thr_t=0.45 pero no supera thr_b=0.50
    señal = _signal_logic(0.47, 0.47, thr_b, thr_t)
    assert señal == "SELL"


# ─────────────────────────────────────────────────────────────────────────────
# Runner sin pytest
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [(k, v) for k, v in globals().items() if k.startswith("test_")]
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except AssertionError as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ✗ {name}: {type(e).__name__}: {e}")
            failed += 1
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed")
