"""
optimize_local_reversal.py — Optimizador de Hiperparámetros del Modelo ML
══════════════════════════════════════════════════════════════════════════════

DIAGNÓSTICO BASE (obtenido analizando prob_bottom.npy vs backtest_irreal.json):
  · El modelo asigna prob_bottom < 0.10 al 80% de los BUY del irreal.
  · Causa principal: la definición de extremo del modelo (_VENTANA_LABEL,
    _VENTANA_FEATURES, precio low/high vs close) no coincide con la del irreal.
  · El irreal opera con ventana=10 sobre low/high exacto; el modelo aprende
    con ventana=12 sobre features de cierre → misalignment estructural.

ESTRATEGIA DE OPTIMIZACIÓN (3 fases, de más rápida a más costosa):
  ──────────────────────────────────────────────────────────────────
  FASE 0 — Alineación sin reentrenamiento (segundos):
    Usa las prob_bottom/prob_top ya cacheadas y barre umbrales thr_b / thr_t.
    Útil para encontrar el máximo rendimiento alcanzable con el modelo actual.

  FASE 1 — Grid sobre parámetros estructurales (minutos-horas):
    Varía VENTANA_LABEL y VENTANA_FEATURES. Cada combo reentrena el modelo.
    Objetivo: alinear la definición de extremo local con la del irreal (ventana≈10).
    Métrica de selección: F1-score entre labels generados y señales del irreal,
    evaluado en O(1) sin backtest completo → filtra candidatos baratos.

  FASE 2 — Fine-tuning de hiperparámetros del GBM (horas):
    Con la mejor config estructural de Fase 1, varía max_depth, learning_rate,
    min_samples_leaf, l2_regularization. Cada combo reentrena y hace backtest.

  FASE 3 — Sweep de umbrales (segundos, sin reentrenamiento):
    Con el mejor modelo de Fase 2, barre thr_b y thr_t en grid fino.

Uso:
    python optimize_local_reversal.py --phase 0
    python optimize_local_reversal.py --phase 1
    python optimize_local_reversal.py --phase 2
    python optimize_local_reversal.py --phase 3
    python optimize_local_reversal.py --all         # fases 0→1→2→3
    python optimize_local_reversal.py --phase 1 --dry-run  # solo F1, sin backtest
    python optimize_local_reversal.py --irreal-json path/to/backtest_results.json

Salida:
    optimize_results_phaseN.json    — resultados de cada fase
    optimize_best.json              — top-5 configuraciones globales
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

# ══════════════════════════════════════════════════════════════════════════════
# ESPACIOS DE BÚSQUEDA
# ══════════════════════════════════════════════════════════════════════════════

# Fase 1: parámetros estructurales
# ventana_label cercana al irreal (10) es el eje más importante
PHASE1_GRID = {
    "ventana_label":    [6, 10, 14, 18],
    "ventana_features": [6, 10, 14, 18],
    "warmup":           [50, 100, 200, 500, 1000],
}

# Fase 2: hiperparámetros del GBM (una vez fijada la config estructural)
PHASE2_GRID = {
    "max_iter":          [150, 300, 600, 1200],
    "max_depth":         [4, 6, 8],
    "learning_rate":     [0.02, 0.05, 0.10],
    "min_samples_leaf":  [10, 15, 25, 40],
    "l2_regularization": [0.0, 0.1, 0.5],
    "class_weight":      ["balanced", None],
}

# Fase 0 y 3: umbrales de señal
THRESHOLD_GRID = {
    "thr_b": np.round(np.arange(0.30, 0.86, 0.05), 3).tolist(),
    "thr_t": np.round(np.arange(0.25, 0.81, 0.05), 3).tolist(),
}

CACHE_BASE = Path(".cache_optimizer")


# ══════════════════════════════════════════════════════════════════════════════
# CARGA DE DATOS DE REFERENCIA (irreal)
# ══════════════════════════════════════════════════════════════════════════════

def load_irreal_signals(json_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga el JSON del irreal y retorna arrays de timestamps (en segundos)
    de señales BUY y SELL ejecutadas (no ignoradas).
    """
    import datetime
    with open(json_path) as f:
        data = json.load(f)

    buy_ts, sell_ts = [], []
    for t in data["trade_history"]:
        if t["ignorado"]:
            continue
        dt = datetime.datetime.fromisoformat(t["datetime"].replace("Z", "+00:00"))
        ts_sec = int(dt.timestamp())
        if t["type"] == "BUY":
            buy_ts.append(ts_sec)
        else:
            sell_ts.append(ts_sec)

    return np.array(buy_ts, dtype=np.int64), np.array(sell_ts, dtype=np.int64)


# ══════════════════════════════════════════════════════════════════════════════
# MÉTRICA DE ALINEACIÓN: F1 entre labels y señales del irreal
# ══════════════════════════════════════════════════════════════════════════════

def label_alignment_score(
    candles,
    ventana_label: int,
    irreal_buy_ts:  np.ndarray,
    irreal_sell_ts: np.ndarray,
    ts_arr:         np.ndarray,
    tolerance_velas: int = 1,
) -> Dict:
    """
    Genera labels con ventana_label y calcula F1-score contra las señales del irreal.

    tolerance_velas: número de velas de margen para considerar un hit.
    Un label bottom[i] es un hit si existe algún BUY del irreal en [i-tol, i+tol].

    Esta métrica es O(N) y no requiere backtest completo → evalúa en milisegundos.
    """
    N   = len(candles)
    VL  = ventana_label

    low  = np.array([c.low  for c in candles])
    high = np.array([c.high for c in candles])

    # Generar labels vectorizados con numpy (mucho más rápido que el loop de Python)
    # bottom: low[i] <= min(low[i-VL:i+VL+1])
    # Usamos stride tricks para ventana deslizante
    pad_low  = np.pad(low,  VL, mode="edge")
    pad_high = np.pad(high, VL, mode="edge")

    # Para cada posición, ventana completa de 2*VL+1
    # Usando np.lib.stride_tricks para eficiencia
    from numpy.lib.stride_tricks import sliding_window_view
    win_low  = sliding_window_view(pad_low,  2 * VL + 1)[:N]
    win_high = sliding_window_view(pad_high, 2 * VL + 1)[:N]

    center = VL
    is_bottom = low  == win_low.min(axis=1)
    is_top    = high == win_high.max(axis=1)

    # Índices de labels generados
    label_bottom_idx = np.where(is_bottom)[0]
    label_top_idx    = np.where(is_top)[0]

    # Mapear timestamps del irreal a índices del array de velas
    ts_to_idx = {int(t): i for i, t in enumerate(ts_arr)}

    buy_idx  = np.array([ts_to_idx[t] for t in irreal_buy_ts  if t in ts_to_idx], dtype=np.int64)
    sell_idx = np.array([ts_to_idx[t] for t in irreal_sell_ts if t in ts_to_idx], dtype=np.int64)

    def compute_f1(label_idx: np.ndarray, irreal_idx: np.ndarray) -> Tuple[float, float, float]:
        """F1 con tolerancia de ±tolerance_velas."""
        if len(irreal_idx) == 0 or len(label_idx) == 0:
            return 0.0, 0.0, 0.0

        # Para cada señal irreal, ¿hay algún label en su vecindad?
        irreal_min = irreal_idx[:, None] - tolerance_velas  # (M,1)
        irreal_max = irreal_idx[:, None] + tolerance_velas
        label_row  = label_idx[None, :]                     # (1,K)

        hits_irreal = np.any(
            (label_row >= irreal_min) & (label_row <= irreal_max),
            axis=1
        )
        recall = hits_irreal.mean()

        # Para cada label, ¿hay alguna señal irreal en su vecindad?
        label_min   = label_idx[:, None] - tolerance_velas
        label_max   = label_idx[:, None] + tolerance_velas
        irreal_row  = irreal_idx[None, :]

        hits_label = np.any(
            (irreal_row >= label_min) & (irreal_row <= label_max),
            axis=1
        )
        precision = hits_label.mean()

        f1 = (2 * precision * recall / (precision + recall + 1e-9))
        return float(f1), float(precision), float(recall)

    f1_b, prec_b, rec_b = compute_f1(label_bottom_idx, buy_idx)
    f1_t, prec_t, rec_t = compute_f1(label_top_idx,    sell_idx)

    return {
        "ventana_label":    ventana_label,
        "n_labels_bottom":  int(is_bottom.sum()),
        "n_labels_top":     int(is_top.sum()),
        "n_irreal_buy":     len(buy_idx),
        "n_irreal_sell":    len(sell_idx),
        "f1_bottom":        round(f1_b, 4),
        "precision_bottom": round(prec_b, 4),
        "recall_bottom":    round(rec_b, 4),
        "f1_top":           round(f1_t, 4),
        "precision_top":    round(prec_t, 4),
        "recall_top":       round(rec_t, 4),
        "f1_combined":      round((f1_b + f1_t) / 2, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER DE UN BACKTEST (reutiliza la estrategia con modelo ya entrenado)
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(strategy, thr_b: float, thr_t: float) -> Dict:
    """
    Corre un backtest completo con la estrategia ya inicializada.
    Solo cambia los umbrales → sin reentrenamiento.
    Retorna métricas del resultado.
    """
    import config_local as CL
    from actors.price_feed   import SQLiteFeed
    from actors.wallet       import JSONWallet, TradeRecord
    from actors.order_book   import SimulatedOrderBook, OrderSide
    from actors.clock        import LocalClock
    from risk.risk_manager   import RiskManager, RiskConfig
    from strategies.base_strategy import SignalSide

    t0 = time.time()
    strategy.thr_b = thr_b
    strategy.thr_t = thr_t
    strategy._candles_seen = 0

    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = "/tmp/_opt_result.json",
    )
    ob    = SimulatedOrderBook(commission_pct=CL.COMMISSION_PCT,
                               max_posiciones=CL.MAX_POSICIONES)
    risk  = RiskManager(config=RiskConfig.permissive(),
                        usdt_inicial=CL.SALDO_USDT_INICIAL)
    clock = LocalClock(feed=feed, start=CL.FECHA_INICIO,
                       end=CL.FECHA_FIN, symbol=CL.SYMBOL)

    n_buy = n_sell = 0
    wins = 0
    buy_prices: List[float] = []
    last_candle = None

    for candle in clock:
        last_candle = candle
        signal = strategy._tick(candle, wallet)
        if not signal.is_actionable:
            continue
        order_side  = signal.to_order_side()
        risk_reason = risk.check(order_side, signal.price, wallet, candle)
        if risk_reason:
            wallet.update(TradeRecord(ts=candle.ts, side=order_side.value,
                                      price=signal.price, ignored=True,
                                      ignore_reason=risk_reason))
            continue
        order = ob.execute_with_guards(order_side, signal.price, wallet,
                                       candle_ts=candle.ts)
        if order.is_filled:
            if order_side == OrderSide.BUY:
                n_buy += 1
                buy_prices.append(signal.price)
            else:
                n_sell += 1
                if buy_prices:
                    if signal.price > sum(buy_prices) / len(buy_prices):
                        wins += 1
                    buy_prices.clear()
        risk.update_peak(wallet.portfolio_value(candle.close))

    if last_candle is None:
        return {"error": "sin_velas"}

    # Buy & Hold
    first = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    p_ini = first[0].close if first else last_candle.close
    bh    = (last_candle.close / p_ini - 1) * 100

    port   = wallet.portfolio_value(last_candle.close)
    pnl    = (port / CL.SALDO_USDT_INICIAL - 1) * 100
    alpha  = pnl - bh
    wr     = wins / max(n_sell, 1)

    return {
        "pnl_pct":       round(pnl, 4),
        "bh_pnl":        round(bh, 4),
        "alpha_vs_bh":   round(alpha, 4),
        "n_compras":     n_buy,
        "n_ventas":      n_sell,
        "win_rate":      round(wr, 4),
        "elapsed_s":     round(time.time() - t0, 2),
    }


# ══════════════════════════════════════════════════════════════════════════════
# FASE 0: sweep de umbrales sobre modelo cacheado (sin reentrenamiento)
# ══════════════════════════════════════════════════════════════════════════════

def phase0_threshold_sweep(irreal_pnl: float, verbose: bool = True) -> List[Dict]:
    """
    Carga el modelo del cache actual y barre todos los pares (thr_b, thr_t).
    Rápido: no reentrena.
    """
    import config_local as CL
    from actors.price_feed         import SQLiteFeed
    from strategies.local_reversal import LocalReversalStrategy

    print("\n" + "═"*60)
    print("  FASE 0: Sweep de umbrales (modelo cacheado)")
    print("═"*60)

    feed = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)

    strategy = LocalReversalStrategy(
        thr_b=0.50, thr_t=0.45,
        cache_dir=".cache_local_reversal",
        force_recompute=False,
    )
    from actors.wallet import JSONWallet
    wallet_dummy = JSONWallet(usdt_inicial=CL.SALDO_USDT_INICIAL,
                              max_posiciones=CL.MAX_POSICIONES,
                              json_path="/tmp/_dummy.json")
    print("  Cargando modelo desde cache...")
    strategy.on_start(wallet=wallet_dummy, feed=feed,
                      start="2017-01-01", end="2030-01-01", symbol=CL.SYMBOL)
    print("  Modelo listo.")

    combos = list(product(THRESHOLD_GRID["thr_b"], THRESHOLD_GRID["thr_t"]))
    print(f"  Evaluando {len(combos)} combinaciones de umbrales...\n")

    results = []
    best_score = -9999

    for i, (thr_b, thr_t) in enumerate(combos, 1):
        r = run_backtest(strategy, thr_b, thr_t)
        if "error" in r:
            continue

        # Score: maximiza PnL con penalidad por distancia al irreal
        gap   = max(irreal_pnl - r["pnl_pct"], 0)
        score = r["pnl_pct"] * 0.5 + r["alpha_vs_bh"] * 0.3 + r["win_rate"] * 100 * 0.2 - gap * 0.2

        entry = {
            "phase": 0, "thr_b": thr_b, "thr_t": thr_t,
            "score": round(score, 3), **r,
        }
        results.append(entry)

        if score > best_score:
            best_score = score
            if verbose:
                print(f"  [{i:3d}/{len(combos)}] ★ NUEVO MEJOR "
                      f"thr_b={thr_b:.2f} thr_t={thr_t:.2f} → "
                      f"PnL={r['pnl_pct']:+.1f}%  alpha={r['alpha_vs_bh']:+.1f}%  "
                      f"wr={r['win_rate']:.1%}  score={score:.2f}")
        elif verbose and i % 20 == 0:
            print(f"  [{i:3d}/{len(combos)}] thr_b={thr_b:.2f} thr_t={thr_t:.2f} → "
                  f"PnL={r['pnl_pct']:+.1f}%  score={score:.2f}")

    results.sort(key=lambda x: x["score"], reverse=True)
    _save_results(results, "phase0")
    _print_top5(results, "FASE 0")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FASE 1: alineación estructural con F1-score (sin backtest completo)
# ══════════════════════════════════════════════════════════════════════════════

def phase1_structural_alignment(
    irreal_json: str,
    dry_run: bool = False,
    verbose: bool = True,
) -> List[Dict]:
    """
    Para cada (ventana_label, ventana_features) del grid:
      1. Calcula F1-score de los labels generados vs señales del irreal (instantáneo).
      2. Si dry_run=False y F1 supera un umbral mínimo: reentrena modelo + backtest.

    Esto filtra el espacio de búsqueda masivamente: de N² backtests a solo los
    candidatos con buena alineación conceptual.
    """
    import config_local as CL
    from actors.price_feed         import SQLiteFeed
    from strategies.local_reversal import LocalReversalStrategy
    from actors.wallet             import JSONWallet

    print("\n" + "═"*60)
    print("  FASE 1: Alineación estructural (ventana_label × ventana_features)")
    print("═"*60)

    irreal_buy_ts, irreal_sell_ts = load_irreal_signals(irreal_json)
    print(f"  Señales irreal: {len(irreal_buy_ts)} BUY  {len(irreal_sell_ts)} SELL")

    feed = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    print("  Cargando velas completas para cálculo de F1...")
    all_candles = feed.get_candles("2017-01-01", "2030-01-01", CL.SYMBOL)
    ts_arr = np.array([c.ts for c in all_candles], dtype=np.int64)
    print(f"  {len(all_candles):,} velas cargadas.")

    combos_vl  = PHASE1_GRID["ventana_label"]
    combos_vf  = PHASE1_GRID["ventana_features"]
    combos_wu  = PHASE1_GRID["warmup"]

    # ── PASO 1: F1-score para todos los ventana_label (instantáneo) ──────────
    print("\n  Paso 1 — Calculando F1-score por ventana_label...")
    f1_results = []
    for vl in combos_vl:
        t0 = time.time()
        r  = label_alignment_score(all_candles, vl, irreal_buy_ts, irreal_sell_ts, ts_arr)
        elapsed = time.time() - t0
        f1_results.append(r)
        print(f"    ventana_label={vl:2d}  "
              f"F1_bottom={r['f1_bottom']:.3f}  F1_top={r['f1_top']:.3f}  "
              f"F1_combined={r['f1_combined']:.3f}  "
              f"labels_B={r['n_labels_bottom']:,}  labels_T={r['n_labels_top']:,}  "
              f"({elapsed:.2f}s)")

    f1_results.sort(key=lambda x: x["f1_combined"], reverse=True)
    best_vl = f1_results[0]["ventana_label"]
    print(f"\n  Mejor ventana_label por F1: {best_vl}")

    if dry_run:
        print("  [dry-run] Saltando backtest completo.")
        _save_results(f1_results, "phase1_f1")
        return f1_results

    # ── PASO 2: backtest para los top-3 ventana_label × todos ventana_features ──
    top_vl = [r["ventana_label"] for r in f1_results[:3]]
    combos  = list(product(top_vl, combos_vf, combos_wu))
    total   = len(combos)
    print(f"\n  Paso 2 — Backtest para top-3 ventana_label × {len(combos_vf)} features × {len(combos_wu)} warmup = {total} combos")
    print(f"  (cada combo reentrena el modelo desde cero — puede tardar varios minutos)\n")

    # Umbrales fijos (calibrados) para comparar solo el efecto estructural
    THR_B_DEFAULT, THR_T_DEFAULT = 0.50, 0.45

    results_bt = []
    best_score = -9999

    for i, (vl, vf, wu) in enumerate(combos, 1):
        cache_dir = CACHE_BASE / f"vl{vl}_vf{vf}_wu{wu}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [{i:2d}/{total}] ventana_label={vl}  ventana_features={vf}  warmup={wu}", end=" → ")

        t_train = time.time()

        # Parchear los parámetros de clase (sin subclasificar)
        LocalReversalStrategy._VENTANA_LABEL    = vl
        LocalReversalStrategy._VENTANA_FEATURES = vf
        LocalReversalStrategy._WARMUP           = wu
        LocalReversalStrategy._N_FEATURES       = vf * 8 + 5

        strategy = LocalReversalStrategy(
            thr_b=THR_B_DEFAULT, thr_t=THR_T_DEFAULT,
            cache_dir=str(cache_dir),
            force_recompute=True,
        )
        wallet_dummy = JSONWallet(
            usdt_inicial=CL.SALDO_USDT_INICIAL,
            max_posiciones=CL.MAX_POSICIONES,
            json_path="/tmp/_dummy.json",
        )
        try:
            strategy.on_start(wallet=wallet_dummy, feed=feed,
                              start="2017-01-01", end="2030-01-01", symbol=CL.SYMBOL)
        except Exception as e:
            print(f"ERROR en entrenamiento: {e}")
            continue

        t_bt  = time.time()
        bt    = run_backtest(strategy, THR_B_DEFAULT, THR_T_DEFAULT)
        if "error" in bt:
            print(f"ERROR en backtest: {bt['error']}")
            continue

        score = bt["pnl_pct"] * 0.5 + bt["alpha_vs_bh"] * 0.3 + bt["win_rate"] * 100 * 0.2
        entry = {
            "phase": 1,
            "ventana_label": vl, "ventana_features": vf, "warmup": wu,
            "thr_b": THR_B_DEFAULT, "thr_t": THR_T_DEFAULT,
            "score": round(score, 3),
            "elapsed_train_s": round(t_bt - t_train, 1),
            **bt,
        }
        results_bt.append(entry)

        marker = "★" if score > best_score else " "
        if score > best_score:
            best_score = score
        print(f"{marker} PnL={bt['pnl_pct']:+.1f}%  alpha={bt['alpha_vs_bh']:+.1f}%  "
              f"wr={bt['win_rate']:.1%}  train={t_bt-t_train:.0f}s")

    results_bt.sort(key=lambda x: x["score"], reverse=True)
    _save_results(results_bt, "phase1")
    _print_top5(results_bt, "FASE 1")
    return results_bt


# ══════════════════════════════════════════════════════════════════════════════
# FASE 2: fine-tuning de hiperparámetros GBM
# ══════════════════════════════════════════════════════════════════════════════

def phase2_gbm_tuning(
    best_structural: Dict,
    n_random: int = 30,
    verbose: bool = True,
) -> List[Dict]:
    """
    Fija la config estructural del mejor resultado de Fase 1 y hace
    random search sobre los hiperparámetros del GBM.

    n_random: número de combinaciones aleatorias a evaluar.
    """
    import random
    import config_local as CL
    from actors.price_feed         import SQLiteFeed
    from strategies.local_reversal import LocalReversalStrategy
    from actors.wallet             import JSONWallet

    vl = best_structural["ventana_label"]
    vf = best_structural["ventana_features"]
    wu = best_structural.get("warmup", 1000)

    print("\n" + "═"*60)
    print(f"  FASE 2: Fine-tuning GBM (vl={vl}, vf={vf}, wu={wu})")
    print("═"*60)

    feed = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)

    # Aplicar config estructural
    LocalReversalStrategy._VENTANA_LABEL    = vl
    LocalReversalStrategy._VENTANA_FEATURES = vf
    LocalReversalStrategy._WARMUP           = wu
    LocalReversalStrategy._N_FEATURES       = vf * 8 + 5

    # Generar n_random combinaciones aleatorias del espacio de GBM
    random.seed(42)
    space = PHASE2_GRID
    combos = []
    for _ in range(n_random):
        combos.append({
            "max_iter":          random.choice(space["max_iter"]),
            "max_depth":         random.choice(space["max_depth"]),
            "learning_rate":     random.choice(space["learning_rate"]),
            "min_samples_leaf":  random.choice(space["min_samples_leaf"]),
            "l2_regularization": random.choice(space["l2_regularization"]),
            "class_weight":      random.choice(space["class_weight"]),
            "random_state":      42,
        })

    # Siempre incluir la config actual como baseline
    combos.insert(0, dict(
        max_iter=400, max_depth=6, learning_rate=0.05,
        min_samples_leaf=15, l2_regularization=0.1,
        class_weight="balanced", random_state=42,
    ))

    print(f"  Evaluando {len(combos)} configs de GBM (random search + baseline)...\n")

    THR_B, THR_T = 0.50, 0.45
    results, best_score = [], -9999

    for i, params in enumerate(combos, 1):
        cache_dir = CACHE_BASE / f"p2_vl{vl}_vf{vf}_{i}"
        cache_dir.mkdir(parents=True, exist_ok=True)

        label = "baseline" if i == 1 else f"combo_{i}"
        print(f"  [{i:2d}/{len(combos)}] {label}  ", end="", flush=True)

        # Parchear _MODEL_PARAMS
        LocalReversalStrategy._MODEL_PARAMS = params

        strategy = LocalReversalStrategy(
            thr_b=THR_B, thr_t=THR_T,
            cache_dir=str(cache_dir),
            force_recompute=True,
        )
        wallet_dummy = JSONWallet(
            usdt_inicial=CL.SALDO_USDT_INICIAL,
            max_posiciones=CL.MAX_POSICIONES,
            json_path="/tmp/_dummy.json",
        )
        t0 = time.time()
        try:
            strategy.on_start(wallet=wallet_dummy, feed=feed,
                              start="2017-01-01", end="2030-01-01", symbol=CL.SYMBOL)
        except Exception as e:
            print(f"ERROR train: {e}")
            continue

        t_bt = time.time()
        bt   = run_backtest(strategy, THR_B, THR_T)
        if "error" in bt:
            print(f"ERROR bt: {bt['error']}")
            continue

        score = bt["pnl_pct"] * 0.5 + bt["alpha_vs_bh"] * 0.3 + bt["win_rate"] * 100 * 0.2
        entry = {
            "phase": 2,
            "ventana_label": vl, "ventana_features": vf, "warmup": wu,
            "model_params": params,
            "thr_b": THR_B, "thr_t": THR_T,
            "score": round(score, 3),
            "elapsed_train_s": round(t_bt - t0, 1),
            **bt,
        }
        results.append(entry)

        marker = "★" if score > best_score else " "
        if score > best_score:
            best_score = score
        print(f"{marker} PnL={bt['pnl_pct']:+.1f}%  alpha={bt['alpha_vs_bh']:+.1f}%  "
              f"wr={bt['win_rate']:.1%}  train={t_bt-t0:.0f}s")

    results.sort(key=lambda x: x["score"], reverse=True)
    _save_results(results, "phase2")
    _print_top5(results, "FASE 2")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# FASE 3: sweep fino de umbrales sobre el mejor modelo
# ══════════════════════════════════════════════════════════════════════════════

def phase3_threshold_sweep(best_config: Dict, verbose: bool = True) -> List[Dict]:
    """
    Con el mejor modelo de Fase 2 ya cacheado, barre todos los umbrales en grid fino.
    """
    import config_local as CL
    from actors.price_feed         import SQLiteFeed
    from strategies.local_reversal import LocalReversalStrategy
    from actors.wallet             import JSONWallet

    vl    = best_config["ventana_label"]
    vf    = best_config["ventana_features"]
    wu    = best_config.get("warmup", 1000)
    mp    = best_config.get("model_params", LocalReversalStrategy._MODEL_PARAMS.copy())

    print("\n" + "═"*60)
    print(f"  FASE 3: Sweep de umbrales (vl={vl}, vf={vf})")
    print("═"*60)

    # Reaplicar config
    LocalReversalStrategy._VENTANA_LABEL    = vl
    LocalReversalStrategy._VENTANA_FEATURES = vf
    LocalReversalStrategy._WARMUP           = wu
    LocalReversalStrategy._N_FEATURES       = vf * 8 + 5
    LocalReversalStrategy._MODEL_PARAMS     = mp

    feed = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)

    # Cargar desde cache (el modelo ya está entrenado)
    cache_dir = CACHE_BASE / f"p2_vl{vl}_vf{vf}_1"  # cache del baseline de fase2
    if not cache_dir.exists():
        cache_dir = Path(".cache_local_reversal")

    strategy = LocalReversalStrategy(
        thr_b=0.50, thr_t=0.45,
        cache_dir=str(cache_dir),
        force_recompute=False,
    )
    wallet_dummy = JSONWallet(
        usdt_inicial=CL.SALDO_USDT_INICIAL,
        max_posiciones=CL.MAX_POSICIONES,
        json_path="/tmp/_dummy.json",
    )
    print("  Cargando modelo desde cache...")
    strategy.on_start(wallet=wallet_dummy, feed=feed,
                      start="2017-01-01", end="2030-01-01", symbol=CL.SYMBOL)
    print("  Modelo listo.")

    combos = list(product(THRESHOLD_GRID["thr_b"], THRESHOLD_GRID["thr_t"]))
    print(f"  Evaluando {len(combos)} pares de umbrales...\n")

    results, best_score = [], -9999

    for i, (thr_b, thr_t) in enumerate(combos, 1):
        bt = run_backtest(strategy, thr_b, thr_t)
        if "error" in bt:
            continue
        score = bt["pnl_pct"] * 0.5 + bt["alpha_vs_bh"] * 0.3 + bt["win_rate"] * 100 * 0.2
        entry = {
            "phase": 3,
            "ventana_label": vl, "ventana_features": vf, "warmup": wu,
            "thr_b": thr_b, "thr_t": thr_t,
            "score": round(score, 3), **bt,
        }
        results.append(entry)
        if score > best_score:
            best_score = score
            if verbose:
                print(f"  [{i:3d}/{len(combos)}] ★ thr_b={thr_b:.2f} thr_t={thr_t:.2f} → "
                      f"PnL={bt['pnl_pct']:+.1f}%  alpha={bt['alpha_vs_bh']:+.1f}%  "
                      f"wr={bt['win_rate']:.1%}")

    results.sort(key=lambda x: x["score"], reverse=True)
    _save_results(results, "phase3")
    _print_top5(results, "FASE 3")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _save_results(results: List[Dict], tag: str) -> None:
    path = Path(f"optimize_results_{tag}.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Resultados guardados en: {path}")


def _print_top5(results: List[Dict], label: str) -> None:
    print(f"\n  ── TOP-5 {label} ──────────────────────────────────────")
    for i, r in enumerate(results[:5], 1):
        pnl   = r.get("pnl_pct", 0)
        alpha = r.get("alpha_vs_bh", 0)
        wr    = r.get("win_rate", 0)
        vl    = r.get("ventana_label", "?")
        vf    = r.get("ventana_features", "?")
        tb    = r.get("thr_b", "?")
        tt    = r.get("thr_t", "?")
        sc    = r.get("score", 0)
        print(f"  {i}. vl={vl} vf={vf}  thr_b={tb}  thr_t={tt}  "
              f"PnL={pnl:+.1f}%  alpha={alpha:+.1f}%  wr={wr:.1%}  score={sc:.2f}")


def _save_best(all_results: List[Dict]) -> None:
    all_results.sort(key=lambda x: x.get("score", -9999), reverse=True)
    path = Path("optimize_best.json")
    with open(path, "w") as f:
        json.dump(all_results[:5], f, indent=2, default=str)
    print(f"\n  Top-5 global guardado en: {path}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Optimizador de hiperparámetros LocalReversalStrategy"
    )
    parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3],
                        help="Fase a ejecutar (0-3)")
    parser.add_argument("--all", action="store_true",
                        help="Ejecutar todas las fases en secuencia")
    parser.add_argument("--dry-run", action="store_true",
                        help="Fase 1 solo calcula F1, sin backtest")
    parser.add_argument("--irreal-json", default="backtest_results_irreal.json",
                        help="Path al JSON del backtest irreal")
    parser.add_argument("--irreal-pnl", type=float, default=306.97,
                        help="PnL%% del irreal como referencia (default: 306.97)")
    parser.add_argument("--n-random", type=int, default=30,
                        help="Número de combos random en Fase 2")
    parser.add_argument("--quiet", action="store_true",
                        help="Menos output durante la búsqueda")
    args = parser.parse_args()

    verbose = not args.quiet
    all_results: List[Dict] = []

    if args.phase == 0 or args.all:
        r0 = phase0_threshold_sweep(
            irreal_pnl=args.irreal_pnl, verbose=verbose
        )
        all_results.extend(r0)

    if args.phase == 1 or args.all:
        r1 = phase1_structural_alignment(
            irreal_json=args.irreal_json,
            dry_run=args.dry_run,
            verbose=verbose,
        )
        all_results.extend(r for r in r1 if "pnl_pct" in r)

    if args.phase == 2 or args.all:
        # Cargar mejor resultado de fase 1
        p1_path = Path("optimize_results_phase1.json")
        if p1_path.exists():
            with open(p1_path) as f:
                r1_loaded = json.load(f)
            best_p1 = r1_loaded[0]
        elif all_results:
            best_p1 = max(all_results, key=lambda x: x.get("score", -9999))
        else:
            print("  ✗ No hay resultados de Fase 1. Ejecutar --phase 1 primero.")
            sys.exit(1)

        r2 = phase2_gbm_tuning(
            best_structural=best_p1,
            n_random=args.n_random,
            verbose=verbose,
        )
        all_results.extend(r2)

    if args.phase == 3 or args.all:
        # Cargar mejor resultado de fase 2
        p2_path = Path("optimize_results_phase2.json")
        if p2_path.exists():
            with open(p2_path) as f:
                r2_loaded = json.load(f)
            best_p2 = r2_loaded[0]
        elif all_results:
            best_p2 = max(all_results, key=lambda x: x.get("score", -9999))
        else:
            print("  ✗ No hay resultados de Fase 2. Ejecutar --phase 2 primero.")
            sys.exit(1)

        r3 = phase3_threshold_sweep(best_config=best_p2, verbose=verbose)
        all_results.extend(r3)

    if all_results:
        _save_best(all_results)

    if not (args.phase is not None or args.all):
        parser.print_help()
        print("\n  Ejemplo rápido: python optimize_local_reversal.py --phase 0")
        print("  Ejemplo completo: python optimize_local_reversal.py --all --irreal-json backtest_results_irreal.json")


if __name__ == "__main__":
    main()
