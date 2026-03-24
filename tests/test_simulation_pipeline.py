"""
tests/test_simulation_pipeline.py
══════════════════════════════════
Tests de integración del pipeline de simulación walk-forward.

AUTÓNOMO: si los artefactos de calibración (oos_v3_test.pkl) no existen,
este módulo los genera desde la DB antes de correr los tests.
La primera ejecución tarda ~90s. Las siguientes son instantáneas.

Requiere DB configurada en config_local.py (DB_PATH).

Si la DB no existe, los tests de integración se marcan SKIP con mensaje.

Ejecutar:
    python tests/test_simulation_pipeline.py
    python -m pytest tests/test_simulation_pipeline.py -v
"""

import sys
import os
import time
import numpy as np
from collections import deque
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ─────────────────────────────────────────────────────────────────────────────
# Rutas
# FIX: DB_PATH se lee de config_local en lugar de estar hardcodeada aquí.
# Antes: DB_PATH = r"C:\Users\Bernardo\Documents\..." (duplicada e inconsistente)
# Ahora: una única fuente de verdad — cambiar config_local.py es suficiente.
# ─────────────────────────────────────────────────────────────────────────────

try:
    from config_local import DB_PATH
except ImportError:
    DB_PATH = ""

CACHE_DIR  = Path(__file__).parent.parent / ".cache_tests"
CACHE_PATH = CACHE_DIR / "oos_v3_test.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros exactos del pipeline de calibración (Etapa 4)
# ─────────────────────────────────────────────────────────────────────────────

COMMISSION   = 0.001
MAX_POS      = 5
CAPITAL      = 1000.0
VENTANA      = 18
WIN          = 24
WARMUP       = 200
MODEL_PARAMS = dict(
    max_iter=400, max_depth=6, learning_rate=0.05,
    min_samples_leaf=15, l2_regularization=0.1,
    random_state=42, class_weight='balanced',
)


# ─────────────────────────────────────────────────────────────────────────────
# Simulador — réplica exacta del usado en calibración
# ─────────────────────────────────────────────────────────────────────────────

def simular_wallet(prob_b, prob_t, idx_te, thr_b, thr_t, close_arr):
    usdt       = CAPITAL
    posiciones = deque()
    slot       = usdt / MAX_POS
    sells_ret  = []
    n_buys = n_sells = 0

    for step, gi in enumerate(idx_te):
        ei = gi + 1
        if ei >= len(close_arr):
            continue
        p = close_arr[ei]

        if prob_t[step] >= thr_t:
            if len(posiciones) > 0:
                btc_total = sum(b for _, b in posiciones)
                btc_venta = btc_total / len(posiciones)
                p_entrada = posiciones.popleft()[0]
                usdt     += btc_venta * p * (1 - COMMISSION)
                sells_ret.append((p - p_entrada) / p_entrada)
                n_sells  += 1
                if len(posiciones) == 0:
                    slot = usdt / MAX_POS

        if prob_b[step] >= thr_b:
            if len(posiciones) < MAX_POS and slot <= usdt + 1e-9:
                btc    = (slot * (1 - COMMISSION)) / p
                usdt  -= slot
                posiciones.append((p, btc))
                n_buys += 1

    ultimo_p = close_arr[min(idx_te[-1] + 1, len(close_arr) - 1)]
    for _, btc in posiciones:
        usdt += btc * ultimo_p * (1 - COMMISSION)

    return sells_ret, usdt, n_buys, n_sells


# ─────────────────────────────────────────────────────────────────────────────
# Fixture: genera o carga el pkl OOS (autónomo)
# ─────────────────────────────────────────────────────────────────────────────

_OOS_CACHE   = None
_CLOSE_CACHE = None


def _build_oos_from_db():
    """Genera el diccionario OOS completo desde la DB. ~90s la primera vez."""
    import sqlite3, pickle
    import pandas as pd
    from sklearn.ensemble import HistGradientBoostingClassifier
    import warnings
    warnings.filterwarnings('ignore')

    print("\n  [fixture] Generando artefactos OOS desde la DB (~90s)...")
    print(f"  [fixture] Se guardarán en: {CACHE_PATH}")
    t_total = time.time()

    conn = sqlite3.connect(DB_PATH)
    df   = pd.read_sql(
        "SELECT timestamp, open, high, low, close, volume, "
        "taker_buy_base_volume FROM btc_hourly ORDER BY timestamp ASC",
        conn
    )
    conn.close()
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    close = df['close'].values
    open_ = df['open'].values
    high  = df['high'].values
    low   = df['low'].values
    vol   = df['volume'].values
    taker = df['taker_buy_base_volume'].values
    N     = len(close)

    rng        = np.where(high - low == 0, 1e-9, high - low)
    body_ratio = np.clip((close - open_) / rng, -1, 1)
    lower_wick = np.clip((np.minimum(open_, close) - low) / rng, 0, 1)
    upper_wick = np.clip((high - np.maximum(open_, close)) / rng, 0, 1)
    delta_r    = np.clip(taker / (vol + 1e-9), 0, 1)
    roll48     = pd.Series(rng).rolling(48, min_periods=1).mean().values
    range_rel  = np.clip(rng / (roll48 + 1e-9), 0, 5)
    low_rej    = np.clip((close - low) / rng, 0, 1)
    high_rej   = np.clip((high - close) / rng, 0, 1)

    def rzs(arr, w=200):
        s  = pd.Series(arr)
        m  = s.rolling(w, min_periods=1).mean()
        st = s.rolling(w, min_periods=1).std().fillna(1).replace(0, 1)
        return ((s - m) / st).values

    ret_4h = pd.Series(close).pct_change(4).fillna(0).values
    div    = rzs(delta_r) - rzs(ret_4h)
    fs     = np.column_stack([
        body_ratio, lower_wick, upper_wick, delta_r,
        range_rel, div, low_rej, high_rej,
    ])

    labels = np.zeros(N, dtype=np.int8)
    for i in range(VENTANA, N - VENTANA):
        if all(low[i]  <= low[j]  for j in range(i - VENTANA, i + VENTANA + 1) if j != i):
            labels[i] = 1; continue
        if all(high[i] >= high[j] for j in range(i - VENTANA, i + VENTANA + 1) if j != i):
            labels[i] = 2

    X_l, y_l, i_l = [], [], []
    for i in range(max(WIN, WARMUP), N - VENTANA - 1):
        w     = fs[i - WIN + 1 : i + 1]
        extra = np.array([
            w[:, 0].mean(), w[-3:, 0].mean(), w[-6:, 5].mean(),
            w[-3:, 6].mean(), w[-3:, 7].mean()
        ])
        X_l.append(np.concatenate([w.flatten(), extra]))
        y_l.append(labels[i])
        i_l.append(i)

    X     = np.array(X_l, dtype=np.float32)
    X     = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y     = np.array(y_l, dtype=np.int8)
    idx   = np.array(i_l)
    years = pd.DatetimeIndex(df['datetime'].values[idx]).year

    oos = {}
    for yr in [2021, 2022, 2023, 2024, 2025]:
        tr_m = years < yr
        te_m = years == yr
        if te_m.sum() == 0:
            continue
        Xtr, ytr = X[tr_m], y[tr_m]
        Xte      = X[te_m]
        t_yr = time.time()
        for clase, nombre in [(1, 'B'), (2, 'T')]:
            m = HistGradientBoostingClassifier(**MODEL_PARAMS)
            m.fit(Xtr, (ytr == clase).astype(int))
            prob = m.predict_proba(Xte)[:, 1]
            oos[(yr, nombre)] = (prob, idx[te_m], y[te_m])
        print(f"  [fixture]   {yr}: {te_m.sum():,} muestras  "
              f"B={(y[te_m]==1).sum()} T={(y[te_m]==2).sum()}  "
              f"({time.time()-t_yr:.0f}s)")

    print(f"  [fixture] Completado en {time.time()-t_total:.0f}s — guardando cache...")
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "wb") as f:
        pickle.dump({'oos': oos, 'close': close}, f)
    print()
    return oos, close


def _load_oos():
    """Carga o genera los artefactos OOS. Cachea en memoria para la sesión."""
    global _OOS_CACHE, _CLOSE_CACHE
    if _OOS_CACHE is not None:
        return _OOS_CACHE, _CLOSE_CACHE

    if CACHE_PATH.exists():
        import pickle
        with open(CACHE_PATH, "rb") as f:
            data = pickle.load(f)
        _OOS_CACHE   = data['oos']
        _CLOSE_CACHE = data['close']
        return _OOS_CACHE, _CLOSE_CACHE

    oos, close  = _build_oos_from_db()
    _OOS_CACHE  = oos
    _CLOSE_CACHE = close
    return oos, close


def _db_available():
    return bool(DB_PATH) and os.path.exists(DB_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# TESTS MATEMÁTICOS — no requieren DB, siempre se ejecutan
# ─────────────────────────────────────────────────────────────────────────────

def test_sin_señales_capital_es_inicial():
    """Umbral 1.0 → ninguna señal → capital final = CAPITAL."""
    n      = 100
    close  = np.full(n, 50_000.0)
    idx    = np.arange(n - 1)
    pb, pt = np.zeros(n - 1), np.zeros(n - 1)
    _, cap, nb, ns = simular_wallet(pb, pt, idx, 1.0, 1.0, close)
    assert nb == 0 and ns == 0
    assert abs(cap - CAPITAL) < 1e-6


def test_buy_sell_mismo_precio_pierde_comisiones():
    """BUY y SELL al mismo precio → pérdida = ~2 comisiones."""
    n     = 100
    close = np.full(n, 50_000.0)
    idx   = np.arange(n - 1)
    pb, pt = np.zeros(n - 1), np.zeros(n - 1)
    pb[10] = pt[20] = 1.0
    _, cap, nb, ns = simular_wallet(pb, pt, idx, 0.99, 0.99, close)
    assert nb == 1 and ns == 1
    assert cap < CAPITAL
    assert 0.30 < (CAPITAL - cap) < 0.60


def test_vender_con_ganancia_incrementa_capital():
    """BUY a 50k, SELL a 55k → capital > inicial."""
    n     = 100
    close = np.concatenate([np.full(50, 50_000.0), np.full(50, 55_000.0)])
    idx   = np.arange(n - 1)
    pb, pt = np.zeros(n - 1), np.zeros(n - 1)
    pb[10] = pt[60] = 1.0
    _, cap, nb, ns = simular_wallet(pb, pt, idx, 0.99, 0.99, close)
    assert cap > CAPITAL and nb == 1 and ns == 1


def test_retorno_correcto_5_pct():
    """Compra 50k, venta 52.5k → retorno = +5% exacto."""
    n     = 100
    close = np.concatenate([np.full(50, 50_000.0), np.full(50, 52_500.0)])
    idx   = np.arange(n - 1)
    pb, pt = np.zeros(n - 1), np.zeros(n - 1)
    pb[10] = pt[60] = 1.0
    sells, _, _, _ = simular_wallet(pb, pt, idx, 0.99, 0.99, close)
    assert len(sells) == 1
    assert abs(sells[0] - 0.05) < 1e-6


def test_retorno_negativo():
    """Compra a 52.5k, venta a 50k → retorno negativo."""
    n     = 100
    close = np.concatenate([np.full(50, 52_500.0), np.full(50, 50_000.0)])
    idx   = np.arange(n - 1)
    pb, pt = np.zeros(n - 1), np.zeros(n - 1)
    pb[10] = pt[60] = 1.0
    sells, _, _, _ = simular_wallet(pb, pt, idx, 0.99, 0.99, close)
    assert len(sells) == 1 and sells[0] < 0


def test_sexta_compra_ignorada():
    """5 posiciones abiertas → 6ta señal BUY ignorada."""
    n     = 200
    close = np.full(n, 50_000.0)
    idx   = np.arange(n - 1)
    pb, pt = np.zeros(n - 1), np.zeros(n - 1)
    for i in [10, 20, 30, 40, 50, 60]:
        pb[i] = 1.0
    _, _, nb, _ = simular_wallet(pb, pt, idx, 0.99, 0.99, close)
    assert nb == 5


def test_slot_crece_por_compounding():
    """Capital crece entre ciclos: 2da compra usa slot más grande."""
    n     = 200
    close = np.concatenate([
        np.full(50, 50_000.0),
        np.full(50, 55_000.0),
        np.full(100, 50_000.0),
    ])
    idx   = np.arange(n - 1)
    pb, pt = np.zeros(n - 1), np.zeros(n - 1)
    pb[10] = pt[60] = pb[110] = 1.0
    _, cap, nb, ns = simular_wallet(pb, pt, idx, 0.99, 0.99, close)
    assert nb == 2 and ns == 1 and cap > CAPITAL


# ─────────────────────────────────────────────────────────────────────────────
# TESTS DE INTEGRACIÓN — requieren DB (autogeneran el pkl si no existe)
# ─────────────────────────────────────────────────────────────────────────────

def test_todos_los_años_oos_son_positivos():
    """
    TEST DE REGRESIÓN PRINCIPAL.

    Con thr_b=0.50, thr_t=0.45 (parámetros calibrados),
    los retornos OOS 2021-2025 deben ser todos positivos.
    """
    if not _db_available():
        print(f"  SKIP: DB no encontrada en {DB_PATH!r}")
        return

    oos, close = _load_oos()
    if oos is None:
        print("  SKIP: no se pudo generar OOS")
        return

    thr_b, thr_t = 0.50, 0.45
    retornos     = {}

    for yr in [2021, 2022, 2023, 2024, 2025]:
        if (yr, 'B') not in oos:
            continue
        pb, idx_te, _ = oos[(yr, 'B')]
        pt, _, _      = oos[(yr, 'T')]
        _, cap, _, _  = simular_wallet(pb, pt, idx_te, thr_b, thr_t, close)
        retornos[yr]  = cap / CAPITAL - 1

    print("  Retornos OOS: " +
          ", ".join(f"{yr}={ret:.1%}" for yr, ret in sorted(retornos.items())))

    for yr, ret in retornos.items():
        assert ret > 0, (
            f"REGRESIÓN: Año {yr} retorno {ret:.1%} — esperado positivo."
        )


def test_estrategia_supera_buy_hold_en_2022():
    """2022: BTC -64%. La estrategia debe ser positiva y superar Buy&Hold."""
    if not _db_available():
        print(f"  SKIP: DB no encontrada en {DB_PATH!r}")
        return

    oos, close = _load_oos()
    if oos is None or (2022, 'B') not in oos:
        print("  SKIP: año 2022 no disponible")
        return

    import sqlite3
    import pandas as pd
    pb, idx_te, _ = oos[(2022, 'B')]
    pt, _, _      = oos[(2022, 'T')]
    _, cap, _, _  = simular_wallet(pb, pt, idx_te, 0.50, 0.45, close)
    ret_estrategia = cap / CAPITAL - 1

    conn  = sqlite3.connect(DB_PATH)
    df_22 = pd.read_sql(
        "SELECT close FROM btc_hourly "
        "WHERE datetime >= '2022-01-01' AND datetime < '2023-01-01' "
        "ORDER BY timestamp ASC",
        conn
    )
    conn.close()
    bh_2022 = df_22['close'].iloc[-1] / df_22['close'].iloc[0] - 1

    print(f"  2022 → Estrategia: {ret_estrategia:+.1%}   "
          f"Buy&Hold: {bh_2022:+.1%}   "
          f"Alpha: {ret_estrategia - bh_2022:+.1%}")

    assert ret_estrategia > 0, \
        f"Estrategia perdió {ret_estrategia:.1%} en 2022 — esperado positivo"
    assert ret_estrategia > bh_2022, \
        f"Estrategia ({ret_estrategia:.1%}) no superó BH ({bh_2022:.1%}) en 2022"


def test_win_rate_mayor_55_pct_en_todos_los_años():
    """Win rate > 55% en cada año OOS."""
    if not _db_available():
        print(f"  SKIP: DB no encontrada en {DB_PATH!r}")
        return

    oos, close = _load_oos()
    if oos is None:
        return

    thr_b, thr_t = 0.50, 0.45
    for yr in [2021, 2022, 2023, 2024, 2025]:
        if (yr, 'B') not in oos:
            continue
        pb, idx_te, _ = oos[(yr, 'B')]
        pt, _, _      = oos[(yr, 'T')]
        sells, _, _, _ = simular_wallet(pb, pt, idx_te, thr_b, thr_t, close)
        if not sells:
            continue
        wr = (np.array(sells) > 0).mean()
        print(f"  WR {yr}: {wr:.1%}")
        assert wr > 0.55, \
            f"Win rate {yr} = {wr:.1%} — por debajo del mínimo (55%)"


def test_2025_positivo_out_of_sample():
    """2025 fue reservado desde el inicio, nunca tocado en calibración."""
    if not _db_available():
        print(f"  SKIP: DB no encontrada en {DB_PATH!r}")
        return

    oos, close = _load_oos()
    if oos is None or (2025, 'B') not in oos:
        print("  SKIP: año 2025 no disponible en los datos")
        return

    pb, idx_te, _ = oos[(2025, 'B')]
    pt, _, _      = oos[(2025, 'T')]
    _, cap, nb, ns = simular_wallet(pb, pt, idx_te, 0.50, 0.45, close)
    ret_2025 = cap / CAPITAL - 1

    print(f"  2025 OOS: {ret_2025:+.1%}  (compras={nb}, ventas={ns})")
    assert ret_2025 > 0, \
        f"2025 retorno {ret_2025:.1%} — esperado positivo (test final OOS)"


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
    print(f"\n{'='*55}")
    print(f"  {passed} passed, {failed} failed")