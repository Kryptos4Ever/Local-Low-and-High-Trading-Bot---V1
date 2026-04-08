"""
backtest_divergence_field.py — Runner + Optimizador de DivergenceFieldStrategy
═══════════════════════════════════════════════════════════════════════════════
Modos de uso:

  python backtest_divergence_field.py
      Corre el backtest con la config por defecto (o la que edites en CONFIG).
      Guarda TE/CMI/field/sink por vela en el JSON para el grid posterior.

  python backtest_divergence_field.py --grid
      FASE 1 — Fast grid: barre score_thresholds y cooldowns sobre los
               valores guardados en el JSON (sin re-computar TE/CMI).
               ~64–200 combinaciones, segundos.

  python backtest_divergence_field.py --deep-grid
      FASE 2 — Deep grid: barre TODOS los hiperparámetros de cómputo
               (estimador, ventana, campo, CMI, normalización, sink).
               Re-corre el backtest completo para cada config.
               ~192 configs × fast grid = miles de combinaciones.
               ⚠ Puede tomar 5–30 minutos según hardware y estimadores.

  python backtest_divergence_field.py --deep-grid --no-knn
      Igual que --deep-grid pero excluye KNN (más rápido).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUÍA DE PARÁMETROS (editar antes de correr)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TE_ESTIMATOR     binning | kde | knn
WINDOW_MODE      fixed | adaptive
WINDOW_SIZE      ventana base (10–30)
FIELD_DEF        analogical | jacobian
CMI_REGIMES      2 | 3
THRESHOLD_MODE   adaptive_percentile | fixed
SINK_MODE        filter_and | score_component
SCORE_THR_BOT    score mínimo para señal BUY  [0.4 – 0.8]
SCORE_THR_TOP    score mínimo para señal SELL [0.4 – 0.8]
COOLDOWN         velas mínimas entre señales  (0 = desactivado)

Referencia estadística (factors_analysis.json, 2021-2025):
  close_position   AUC BOT=0.825  TOP=0.822  → es el proxy más cercano al TE/field
  Esperado para TE/CMI: AUC tentativo 0.60–0.75 (validar con análisis factorial)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import replace
from itertools import product
from pathlib   import Path
from typing    import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL

from actors.price_feed    import SQLiteFeed
from actors.wallet        import JSONWallet, MemoryWallet, TradeRecord
from actors.order_book    import SimulatedOrderBook, OrderSide
from actors.clock         import LocalClock
from risk.risk_manager    import RiskManager, RiskConfig
from state.state_manager  import MemoryStateManager, Checkpoint
from strategies.divergence_field_strategy import (
    DivergenceFieldStrategy, DFConfig,
    TEEstimator, WindowMode, FieldDefinition,
    CMIRegimes, ThresholdMode, SinkMode,
)
from strategies.base_strategy import SignalSide
from support.logger import get_logger

log = get_logger("backtest_divfield")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — Editar aquí antes de cada ejecución
# ══════════════════════════════════════════════════════════════════════════════

CONFIG = DFConfig(
    te_estimator        = TEEstimator.BINNING,
    window_mode         = WindowMode.FIXED,
    window_size         = 20,
    field_def           = FieldDefinition.ANALOGICAL,
    cmi_regimes         = CMIRegimes.BINARY,
    threshold_mode      = ThresholdMode.ADAPTIVE_PERCENTILE,
    sink_mode           = SinkMode.SCORE_COMPONENT,
    score_threshold_bot = 0.55,
    score_threshold_top = 0.55,
    cooldown            = 0,
    k_bins              = 4,
    k_nn                = 3,
    n_norm              = 200,
)

RESULTS_JSON = CL.RESULTS_JSON

# ── Parámetros del fast grid ──────────────────────────────────────────────────
GRID_SCORE_BOT   = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
GRID_SCORE_TOP   = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
GRID_COOLDOWNS   = [0, 24, 48, 96]

# ── Parámetros del deep grid ──────────────────────────────────────────────────
DEEP_TE_ESTIMATORS  = [TEEstimator.BINNING, TEEstimator.KDE, TEEstimator.KNN]
DEEP_WINDOWS        = [10, 15, 20, 30]
DEEP_FIELD_DEFS     = [FieldDefinition.ANALOGICAL, FieldDefinition.JACOBIAN]
DEEP_CMI_REGIMES    = [CMIRegimes.BINARY, CMIRegimes.TERNARY]
DEEP_THRESHOLD_MODES= [ThresholdMode.ADAPTIVE_PERCENTILE, ThresholdMode.FIXED]
DEEP_SINK_MODES     = [SinkMode.SCORE_COMPONENT, SinkMode.FILTER_AND]

GRID_CSV         = "grid_divfield_results.csv"
GRID_JSON_OUT    = "grid_divfield_results.json"
DEEP_CSV         = "deep_grid_divfield_results.csv"
DEEP_JSON_OUT    = "deep_grid_divfield_results.json"


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: enriquecer trade_log con todos los features calculados
# ══════════════════════════════════════════════════════════════════════════════

def _enrich(wallet: JSONWallet, strat: DivergenceFieldStrategy) -> None:
    """Añade todos los features del último trade para el grid posterior."""
    entries = wallet.get_trade_log()
    if not entries:
        return
    last = entries[-1]
    last["te_raw"]           = strat.last_te
    last["cmi_raw"]          = strat.last_cmi
    last["field_price_div"]  = strat.last_field_price
    last["field_vol_div"]    = strat.last_field_vol
    last["field_curl"]       = strat.last_field_curl
    last["sink_raw"]         = strat.last_sink
    last["te_norm"]          = strat.last_te_norm
    last["cmi_norm"]         = strat.last_cmi_norm
    last["field_norm"]       = strat.last_field_norm
    last["sink_norm"]        = strat.last_sink_norm
    last["score_bot"]        = strat.last_score_bot
    last["score_top"]        = strat.last_score_top
    last["is_bot_pattern"]   = strat.last_is_bot_pattern
    last["is_top_pattern"]   = strat.last_is_top_pattern


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    config:       DFConfig,
    results_json: str      = None,
    silent:       bool     = False,
) -> dict:
    """
    Corre el backtest completo con la config dada.
    Retorna el summary dict.
    Escribe JSON si results_json no es None.
    """
    rjson = results_json or RESULTS_JSON
    t_start = time.time()

    cfg_d = config.to_dict()
    if not silent:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║     BACKTEST DIVERGENCE FIELD — Info. Teórica BTC/USDT  ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Rango          : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
        print(f"  Capital        : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
        print(f"  Max posiciones : {CL.MAX_POSICIONES}")
        print(f"  Comisión       : {CL.COMMISSION_PCT}%")
        print(f"  TE estimador   : {cfg_d['te_estimator']}")
        print(f"  Ventana        : {cfg_d['window_size']} ({cfg_d['window_mode']})")
        print(f"  Campo          : {cfg_d['field_def']}")
        print(f"  CMI regímenes  : {cfg_d['cmi_regimes']}")
        print(f"  Umbral mode    : {cfg_d['threshold_mode']}")
        print(f"  Sink mode      : {cfg_d['sink_mode']}")
        print(f"  Score BUY/SELL : {cfg_d['score_threshold_bot']} / {cfg_d['score_threshold_top']}")
        print(f"  Cooldown       : {cfg_d['cooldown']} velas" if cfg_d['cooldown']
              else f"  Cooldown       : desactivado")
        print(f"  Output JSON    : {rjson}")
        print("─" * 60)

    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    clock  = LocalClock(feed=feed, start=CL.FECHA_INICIO, end=CL.FECHA_FIN,
                        symbol=CL.SYMBOL)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = rjson,
    )
    ob    = SimulatedOrderBook(commission_pct=CL.COMMISSION_PCT,
                               max_posiciones=CL.MAX_POSICIONES)
    risk  = RiskManager(config=RiskConfig.permissive(),
                        usdt_inicial=CL.SALDO_USDT_INICIAL)
    state = MemoryStateManager()

    strategy = DivergenceFieldStrategy(config)
    strategy.on_start(wallet)

    n_compras = n_ventas = n_ignorados = 0
    ign_motivos: Dict[str, int] = {}
    precio_min = float("inf")
    precio_max = float("-inf")
    last_candle = None

    if not silent:
        print("Procesando velas...", end=" ", flush=True)

    for candle in clock:
        last_candle = candle
        signal      = strategy._tick(candle, wallet)

        if not signal.is_actionable:
            continue

        order_side  = signal.to_order_side()
        risk_reason = risk.check(order_side, signal.price, wallet, candle)

        if risk_reason:
            n_ignorados += 1
            ign_motivos[risk_reason] = ign_motivos.get(risk_reason, 0) + 1
            wallet.update(TradeRecord(
                ts=candle.ts, side=order_side.value, price=signal.price,
                ignored=True, ignore_reason=risk_reason,
            ))
            _enrich(wallet, strategy)
            continue

        order = ob.execute_with_guards(
            order_side, signal.price, wallet, candle_ts=candle.ts
        )
        _enrich(wallet, strategy)

        if order.is_filled:
            if order_side == OrderSide.BUY:
                n_compras += 1
                precio_min = min(precio_min, signal.price)
            else:
                n_ventas += 1
                precio_max = max(precio_max, signal.price)
        else:
            n_ignorados += 1
            motivo = order.reject_reason or "desconocido"
            ign_motivos[motivo] = ign_motivos.get(motivo, 0) + 1

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(
            wallet, candle.ts, candle.close,
            metadata={"estrategia": strategy.name},
        ))

    if not silent:
        print(f"OK  ({clock.total_candles:,} velas)")

    strategy.on_stop(wallet)

    if last_candle is None:
        if not silent:
            print("✗ Sin velas en el rango indicado.")
        return {}

    precio_final   = last_candle.close
    port_final     = wallet.portfolio_value(precio_final)
    pnl_pct        = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100
    first_candles  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_inicial = first_candles[0].close if first_candles else precio_final
    bh_pnl         = (precio_final / precio_inicial - 1) * 100

    summary = {
        "estrategia":              strategy.name,
        "fecha_inicio":            CL.FECHA_INICIO,
        "fecha_fin":               CL.FECHA_FIN,
        "saldo_inicial_usdt":      CL.SALDO_USDT_INICIAL,
        "usdt_balance_final":      round(wallet.get_usdt_balance(), 8),
        "btc_balance_final":       round(wallet.get_btc_balance(), 10),
        "btc_acumulado_total":     round(wallet.get_btc_acumulado(), 10),
        "btc_en_posiciones_final": round(wallet.btc_en_posiciones(), 10),
        "precio_promedio_final":   round(wallet.precio_promedio_posiciones(), 8),
        "portfolio_value_final":   round(port_final, 4),
        "pnl_pct":                 round(pnl_pct, 4),
        "buy_hold_pnl_pct":        round(bh_pnl, 4),
        "alpha_vs_bh":             round(pnl_pct - bh_pnl, 4),
        "precio_min_comprado":     round(precio_min, 4) if n_compras else None,
        "precio_max_vendido":      round(precio_max, 4) if n_ventas  else None,
        "total_trades_ejecutados": n_compras + n_ventas,
        "total_compras":           n_compras,
        "total_ventas":            n_ventas,
        "total_ignorados":         n_ignorados,
        "ignorados_por_motivo":    ign_motivos,
        "positions_count_final":   wallet.positions_count,
        "usdt_reserva_aplicada":   0.0,
        "umbral_filtro":           config.score_threshold_bot,
        "ordenes_canceladas":      0,
        "parametros": {
            **strategy.describe(),
            "max_posiciones":  CL.MAX_POSICIONES,
            "commission_pct":  CL.COMMISSION_PCT,
            "slot_usdt_final": round(wallet.get_slot_usdt(), 4),
        },
    }

    wallet.flush(summary)

    elapsed = time.time() - t_start
    if not silent:
        sep  = "═" * 60
        sign = "+" if pnl_pct >= 0 else ""
        print(f"\n{sep}")
        print("  RESUMEN")
        print(sep)
        print(f"  Portfolio final  : ${port_final:>12,.2f} USDT")
        print(f"  └─ USDT libre    : ${wallet.get_usdt_balance():>12,.2f}")
        print(f"  └─ BTC en posic. :  {wallet.btc_en_posiciones():.8f} BTC")
        print(f"  PnL              : {sign}{pnl_pct:.2f}%")
        print(f"  Buy & Hold ref   : {bh_pnl:+.2f}%")
        print(f"  Alpha vs B&H     : {pnl_pct - bh_pnl:+.2f}%")
        print(f"  Compras          : {n_compras:,}")
        print(f"  Ventas           : {n_ventas:,}")
        print(f"  Ignorados        : {n_ignorados:,}  → {ign_motivos}")
        print(f"  Posiciones abier.: {wallet.positions_count}")
        print(f"  Tiempo           : {elapsed:.1f}s")
        print(sep)
        print(f"\n✓ Resultado guardado en: {rjson}")

    return summary


# ══════════════════════════════════════════════════════════════════════════════
# FAST GRID — re-simula con distintos score_thresholds y cooldowns
# ══════════════════════════════════════════════════════════════════════════════

def _simulate_thresholds(
    trades:     List[dict],
    thr_bot:    float,
    thr_top:    float,
    cooldown:   int,
    usdt_ini:   float,
    max_pos:    int,
    commission: float,
    bh_pnl:     float,
) -> dict:
    """
    Re-simula la wallet usando los scores guardados en el JSON.
    No re-computa TE/CMI/field — solo ajusta los umbrales de decisión.

    Los scores (score_bot, score_top) y el patrón del campo (is_bot_pattern,
    is_top_pattern) ya fueron calculados y guardados por run_backtest().
    """
    wallet = MemoryWallet(usdt_ini, max_pos)
    n_buy = n_sell = n_ign = 0
    ganancias: List[float] = []

    _NEG = -(10 ** 9)
    last_bot = _NEG
    last_top = _NEG

    for idx, t in enumerate(trades):
        price  = t.get("price", 0.0)
        ts     = t.get("ts", 0)
        sb     = t.get("score_bot")
        st     = t.get("score_top")
        is_bot = t.get("is_bot_pattern", False)
        is_top = t.get("is_top_pattern", False)

        if price <= 0 or sb is None or st is None:
            continue

        cd_ok_bot = cooldown == 0 or (idx - last_bot) >= cooldown
        cd_ok_top = cooldown == 0 or (idx - last_top) >= cooldown

        # SELL prioridad
        if st >= thr_top and is_top and cd_ok_top:
            last_top = idx
            if wallet.positions_count == 0:
                n_ign += 1
            else:
                bpv = wallet.get_btc_por_venta()
                if bpv <= 0:
                    n_ign += 1
                else:
                    usdt_bruto = bpv * price
                    comm       = usdt_bruto * commission / 100.0
                    usdt_neto  = usdt_bruto - comm
                    ganancia   = usdt_neto - wallet.get_slot_usdt()
                    wallet.update(TradeRecord(
                        ts=ts, side="SELL", price=price,
                        btc_sold=bpv, usdt_received=usdt_neto,
                        commission=comm, ganancia_usdt=ganancia,
                    ))
                    ganancias.append(ganancia)
                    n_sell += 1
            continue

        if sb >= thr_bot and is_bot and cd_ok_bot:
            last_bot = idx
            if wallet.positions_count >= max_pos:
                n_ign += 1
            else:
                slot = wallet.get_slot_usdt()
                if slot > wallet.get_usdt_balance() + 1e-9:
                    n_ign += 1
                else:
                    comm = slot * commission / 100.0
                    btc  = (slot - comm) / price
                    wallet.update(TradeRecord(
                        ts=ts, side="BUY", price=price,
                        usdt_spent=slot, btc_bought=btc, commission=comm,
                    ))
                    n_buy += 1

    last_price = trades[-1].get("price", 0.0) if trades else 0.0
    port  = wallet.portfolio_value(last_price)
    pnl   = (port / usdt_ini - 1) * 100.0
    wr    = (sum(1 for g in ganancias if g > 0) / len(ganancias) * 100.0
             if ganancias else 0.0)

    return {
        "thr_bot":  thr_bot,
        "thr_top":  thr_top,
        "cooldown": cooldown,
        "pnl_pct":  round(pnl, 2),
        "portfolio":round(port, 2),
        "alpha_bh": round(pnl - bh_pnl, 2),
        "n_buy":    n_buy,
        "n_sell":   n_sell,
        "n_trades": n_buy + n_sell,
        "n_ign":    n_ign,
        "win_rate": round(wr, 1),
    }


def fast_grid(json_path: str) -> List[dict]:
    """Fase 1: barre score_thresholds y cooldowns sobre valores guardados."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    FAST GRID — Score Thresholds × Cooldowns             ║")
    print("╚══════════════════════════════════════════════════════════╝")

    json_file = Path(json_path)
    if not json_file.exists():
        print(f"\n✗ No se encontró: {json_path}")
        print("  Ejecutar primero: python backtest_divergence_field.py")
        return []

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    trades_raw = data.get("trade_history", [])
    summary_in = data.get("summary", {})
    orig_pnl   = summary_in.get("pnl_pct", 0.0)
    bh_pnl     = summary_in.get("buy_hold_pnl_pct", 0.0)

    # Filtrar trades con score_bot/top guardados
    trades = [t for t in trades_raw
              if t.get("score_bot") is not None and t.get("score_top") is not None]
    print(f"  JSON           : {json_path}")
    print(f"  Período        : {summary_in.get('fecha_inicio')} → {summary_in.get('fecha_fin')}")
    print(f"  PnL original   : {orig_pnl:+.2f}%   B&H: {bh_pnl:+.2f}%")
    print(f"  Trades con scores: {len(trades):,}")

    if not trades:
        print("\n✗ No hay scores guardados. Re-correr el backtest primero.")
        return []

    usdt_ini   = summary_in.get("saldo_inicial_usdt", CL.SALDO_USDT_INICIAL)
    max_pos    = (summary_in.get("parametros", {}).get("max_posiciones", CL.MAX_POSICIONES))
    commission = CL.COMMISSION_PCT

    combos = list(product(GRID_SCORE_BOT, GRID_SCORE_TOP, GRID_COOLDOWNS))
    print(f"\n  Barriendo {len(GRID_SCORE_BOT)} thr_bot × {len(GRID_SCORE_TOP)} thr_top × "
          f"{len(GRID_COOLDOWNS)} cooldowns = {len(combos)} combinaciones...")

    t0      = time.time()
    results = []

    for thr_b, thr_t, cd in combos:
        r = _simulate_thresholds(trades, thr_b, thr_t, cd,
                                 usdt_ini, max_pos, commission, bh_pnl)
        r["alpha_orig"] = round(r["pnl_pct"] - orig_pnl, 2)
        results.append(r)

    results.sort(key=lambda x: -x["pnl_pct"])
    elapsed = time.time() - t0

    _print_grid_results(results[:20], orig_pnl, bh_pnl, elapsed)
    _save_grid(results, GRID_CSV, GRID_JSON_OUT, summary_in, elapsed)

    print(f"\n✓ Fast grid completado en {elapsed:.1f}s")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# DEEP GRID — barre configuraciones de cómputo + fast grid interno
# ══════════════════════════════════════════════════════════════════════════════

def deep_grid(include_knn: bool = True) -> None:
    """
    Fase 2: barre todas las combinaciones de hiperparámetros de cómputo.
    Para cada config, corre el backtest completo y un mini fast-grid.
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    DEEP GRID — Configuraciones de Cómputo               ║")
    print("╚══════════════════════════════════════════════════════════╝")

    estimators = DEEP_TE_ESTIMATORS if include_knn else \
                 [e for e in DEEP_TE_ESTIMATORS if e != TEEstimator.KNN]

    # Configs de cómputo (ventana mínima por estimador)
    compute_configs = []
    for te, win, fd, cr, tm, sm in product(
        estimators, DEEP_WINDOWS, DEEP_FIELD_DEFS,
        DEEP_CMI_REGIMES, DEEP_THRESHOLD_MODES, DEEP_SINK_MODES
    ):
        # KNN requiere al menos ventana 15
        if te == TEEstimator.KNN and win < 15:
            continue
        compute_configs.append(DFConfig(
            te_estimator   = te,
            window_size    = win,
            field_def      = fd,
            cmi_regimes    = cr,
            threshold_mode = tm,
            sink_mode      = sm,
            # score thresholds: default — el fast-grid los optimizará
            score_threshold_bot = 0.55,
            score_threshold_top = 0.55,
        ))

    total_configs = len(compute_configs)
    fast_combos   = len(GRID_SCORE_BOT) * len(GRID_SCORE_TOP) * len(GRID_COOLDOWNS)
    total_runs    = total_configs * fast_combos

    print(f"  Configs de cómputo : {total_configs}")
    print(f"  Fast grid por config: {fast_combos}")
    print(f"  Total combinaciones: {total_runs:,}")
    print(f"  KNN incluido       : {'Sí' if include_knn else 'No'}")

    knn_warning = (
        "\n  ⚠ KNN activo: las configs con KNN son más lentas (~5–10x)."
        "\n    Añadir --no-knn para excluirlas."
    ) if include_knn else ""
    if knn_warning:
        print(knn_warning)
    print()

    all_results = []
    feed = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    first_candles  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_inicial = first_candles[0].close if first_candles else 0.0

    t_total = time.time()
    import tempfile, os

    for i, cfg in enumerate(compute_configs, 1):
        cfg_label = (f"{cfg.te_estimator.value[:3]}|"
                     f"w{cfg.window_size}|"
                     f"{cfg.field_def.value[:3]}|"
                     f"cmi{int(cfg.cmi_regimes)}|"
                     f"{cfg.threshold_mode.value[:3]}|"
                     f"{cfg.sink_mode.value[:3]}")

        print(f"  [{i:>3}/{total_configs}] {cfg_label}", end="  ", flush=True)
        t_cfg = time.time()

        # Guardar en archivo temporal
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json',
                                         delete=False) as tmp:
            tmp_path = tmp.name

        try:
            summary = run_backtest(cfg, results_json=tmp_path, silent=True)
            if not summary:
                print("sin datos")
                continue

            pnl_orig = summary.get("pnl_pct", 0.0)
            bh_pnl   = summary.get("buy_hold_pnl_pct", 0.0)

            # Cargar trade_history del temporal
            with open(tmp_path, encoding="utf-8") as f:
                tmp_data = json.load(f)
            trades = [t for t in tmp_data.get("trade_history", [])
                      if t.get("score_bot") is not None]

            if not trades:
                print(f"sin señales")
                continue

            usdt_ini = summary.get("saldo_inicial_usdt", CL.SALDO_USDT_INICIAL)
            max_pos  = (summary.get("parametros", {})
                        .get("max_posiciones", CL.MAX_POSICIONES))

            # Fast grid sobre esta config
            best_pnl   = -999.0
            best_combo = None

            for thr_b, thr_t, cd in product(GRID_SCORE_BOT, GRID_SCORE_TOP, GRID_COOLDOWNS):
                r = _simulate_thresholds(
                    trades, thr_b, thr_t, cd,
                    usdt_ini, max_pos, CL.COMMISSION_PCT, bh_pnl
                )
                if r["pnl_pct"] > best_pnl:
                    best_pnl   = r["pnl_pct"]
                    best_combo = r

            if best_combo:
                row = {
                    "te_estimator":  cfg.te_estimator.value,
                    "window_size":   cfg.window_size,
                    "field_def":     cfg.field_def.value,
                    "cmi_regimes":   int(cfg.cmi_regimes),
                    "threshold_mode":cfg.threshold_mode.value,
                    "sink_mode":     cfg.sink_mode.value,
                    "orig_pnl":      round(pnl_orig, 2),
                    "best_pnl":      round(best_pnl, 2),
                    "best_thr_bot":  best_combo["thr_bot"],
                    "best_thr_top":  best_combo["thr_top"],
                    "best_cooldown": best_combo["cooldown"],
                    "alpha_bh":      round(best_pnl - bh_pnl, 2),
                    "n_trades":      best_combo["n_trades"],
                    "win_rate":      best_combo["win_rate"],
                    "elapsed_s":     round(time.time() - t_cfg, 1),
                }
                all_results.append(row)
                pnl_s = f"{'+' if best_pnl>=0 else ''}{best_pnl:.2f}%"
                print(f"PnL={pnl_s:>9}  α={row['alpha_bh']:+.2f}%  "
                      f"B={best_combo['thr_bot']}/S={best_combo['thr_top']}  "
                      f"{time.time()-t_cfg:.1f}s")
            else:
                print("sin resultados")

        except Exception as e:
            print(f"ERROR: {e}")
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass

    elapsed_total = time.time() - t_total
    all_results.sort(key=lambda x: -x["best_pnl"])

    print(f"\n{'═'*60}")
    print(f"  DEEP GRID — TOP-20 RESULTADOS")
    print(f"{'═'*60}")
    sep = "─" * 100
    print(f"  {'#':>2}  {'TE':>7} {'W':>3} {'Field':>5} {'CMI':>3} "
          f"{'ThMod':>5} {'Sink':>5}  "
          f"{'best_PnL':>9} {'α_BH':>8}  "
          f"{'b_B':>5} {'b_S':>5} {'b_CD':>4}  "
          f"{'Trades':>6} {'WR%':>5}")
    print(sep)

    for i, r in enumerate(all_results[:20], 1):
        pnl_s = f"{'+' if r['best_pnl']>=0 else ''}{r['best_pnl']:.2f}%"
        abh_s = f"{'+' if r['alpha_bh']>=0 else ''}{r['alpha_bh']:.2f}%"
        print(f"  {i:>2}.  "
              f"{r['te_estimator']:>7} {r['window_size']:>3} "
              f"{r['field_def'][:5]:>5} {r['cmi_regimes']:>3} "
              f"{r['threshold_mode'][:5]:>5} {r['sink_mode'][:5]:>5}  "
              f"{pnl_s:>9} {abh_s:>9}  "
              f"{r['best_thr_bot']:>5.2f} {r['best_thr_top']:>5.2f} "
              f"{str(r['best_cooldown']):>4}  "
              f"{r['n_trades']:>6} {r['win_rate']:>5.1f}%")
    print(sep)
    print(f"  Total configs: {len(all_results)}  |  Tiempo: {elapsed_total:.1f}s")

    # Guardar deep grid
    if all_results:
        fieldnames = list(all_results[0].keys())
        with open(DEEP_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(all_results)
        with open(DEEP_JSON_OUT, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "total_configs":   total_configs,
                    "fast_combos":     fast_combos,
                    "elapsed_s":       round(elapsed_total, 1),
                    "include_knn":     include_knn,
                },
                "top_20": all_results[:20],
                "all":    all_results,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Deep grid guardado  →  {DEEP_CSV}  |  {DEEP_JSON_OUT}")

    if all_results:
        best = all_results[0]
        print(f"\n  MEJOR CONFIG:")
        print(f"    te_estimator   = TEEstimator.{best['te_estimator'].upper()}")
        print(f"    window_size    = {best['window_size']}")
        print(f"    field_def      = FieldDefinition.{best['field_def'].upper()}")
        print(f"    cmi_regimes    = CMIRegimes.{['BINARY','TERNARY'][best['cmi_regimes']-2]}")
        print(f"    threshold_mode = ThresholdMode.{'ADAPTIVE_PERCENTILE' if best['threshold_mode']=='adaptive_percentile' else 'FIXED'}")
        print(f"    sink_mode      = SinkMode.{'SCORE_COMPONENT' if best['sink_mode']=='score_component' else 'FILTER_AND'}")
        print(f"    score_threshold_bot = {best['best_thr_bot']}")
        print(f"    score_threshold_top = {best['best_thr_top']}")
        print(f"    cooldown            = {best['best_cooldown']}")


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS DE PRESENTACIÓN Y PERSISTENCIA
# ══════════════════════════════════════════════════════════════════════════════

def _print_grid_results(
    results: List[dict], orig_pnl: float, bh_pnl: float, elapsed: float
) -> None:
    sep  = "─" * 78
    sep2 = "═" * 78
    print(f"\n{sep2}")
    print(f"  TOP-{len(results)} COMBINACIONES — ordenadas por PnL")
    print(sep2)
    print(f"  {'#':>2}  {'thr_bot':>7} {'thr_top':>7} {'cd':>5}  "
          f"{'PnL%':>8} {'α_BH':>8} {'α_orig':>8}  {'B/S':>9}  {'WR%':>5}")
    print(sep)
    for i, r in enumerate(results, 1):
        pnl_s  = f"{'+' if r['pnl_pct']>=0 else ''}{r['pnl_pct']:.2f}%"
        abh_s  = f"{'+' if r['alpha_bh']>=0 else ''}{r['alpha_bh']:.2f}%"
        aor_s  = f"{'+' if r['alpha_orig']>=0 else ''}{r['alpha_orig']:.2f}%"
        cd_s   = str(r["cooldown"]) if r["cooldown"] else "off"
        bs_s   = f"{r['n_buy']}B/{r['n_sell']}S"
        print(f"  {i:>2}.  {r['thr_bot']:>7.2f} {r['thr_top']:>7.2f} {cd_s:>5}  "
              f"{pnl_s:>9} {abh_s:>9} {aor_s:>9}  {bs_s:>9}  {r['win_rate']:>5.1f}%")
    print(sep)
    print(f"  B&H: {bh_pnl:+.2f}%   Original: {orig_pnl:+.2f}%   Tiempo: {elapsed:.1f}s")


def _save_grid(results: List[dict], csv_path: str, json_path: str,
               orig_summary: dict, elapsed: float) -> None:
    if not results:
        return
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "meta": {
                "fecha_inicio":  orig_summary.get("fecha_inicio"),
                "fecha_fin":     orig_summary.get("fecha_fin"),
                "pnl_original":  orig_summary.get("pnl_pct"),
                "bh":            orig_summary.get("buy_hold_pnl_pct"),
                "grid_bot":      GRID_SCORE_BOT,
                "grid_top":      GRID_SCORE_TOP,
                "grid_cooldown": GRID_COOLDOWNS,
                "n_combos":      len(results),
                "elapsed_s":     round(elapsed, 2),
            },
            "top_20": results[:20],
            "all":    results,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Grid guardado  →  {csv_path}  |  {json_path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest DivergenceFieldStrategy — TE + CMI + Campo Vectorial",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Flujo recomendado:
  1. python backtest_divergence_field.py
         Corre con config por defecto. Guarda scores en JSON.

  2. python backtest_divergence_field.py --grid
         Fast grid: barre score_thresholds y cooldowns (~segundos).
         Muestra top-20 y guarda CSV/JSON.

  3. python backtest_divergence_field.py --deep-grid --no-knn
         Deep grid sin KNN (~minutos). Barre estimadores, ventanas,
         campos, CMI, normalización y modo sink.
         Muestra la mejor config al final.

  4. python backtest_divergence_field.py --deep-grid
         Deep grid completo incluyendo KNN (~10-30 minutos).

  5. Copiar la mejor config del paso 3/4 en CONFIG al principio del archivo
     y re-correr el paso 1 para el backtest definitivo.
""",
    )
    parser.add_argument("--grid",      action="store_true",
                        help="Fast grid sobre scores guardados")
    parser.add_argument("--deep-grid", action="store_true",
                        help="Deep grid: barre todas las configs de cómputo")
    parser.add_argument("--no-knn",    action="store_true",
                        help="Excluir KNN del deep grid (más rápido)")
    parser.add_argument("--json", default=RESULTS_JSON, metavar="PATH",
                        help=f"JSON para los modos --grid (default: {RESULTS_JSON})")
    args = parser.parse_args()

    if args.deep_grid:
        deep_grid(include_knn=not args.no_knn)
    elif args.grid:
        fast_grid(args.json)
    else:
        run_backtest(CONFIG)
