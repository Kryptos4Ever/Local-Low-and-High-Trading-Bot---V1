"""
backtest_predictive_candles.py — Runner de PredictiveCandles
═════════════════════════════════════════════════════════════
Ejecuta PredictiveCandlesStrategy sobre la DB local y produce un JSON
compatible con Graficador.py.

Uso normal:
    python backtest_predictive_candles.py

Modo grid (analiza combinaciones sobre el trade_history existente):
    python backtest_predictive_candles.py --grid
    python backtest_predictive_candles.py --grid --json otro_archivo.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUÍA DE PARÁMETROS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VENTANA          velas hacia atrás para calcular los factores    [10, 16]
UMBRAL_BOT/TOP   score mínimo para BUY/SELL                      (0, 1]
COOLDOWN_BOT/TOP velas mínimas entre señales del mismo tipo       0=off
USE_BOT_* / USE_TOP_*
                 Booleanos por predictor. Los pesos se renormalizan
                 automáticamente sobre los que estén activos.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODO --grid
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Barre 10,584 combinaciones (7 BOT × 7 TOP × 6 thr × 6 cooldown)
sin re-correr el backtest: usa los pred_* guardados en trade_history.

Salida del grid:
  · Top-15 combinaciones generales ordenadas por PnL
  · Tabla de ranking por predictor individual y por combinación de BOT/TOP
  · grid_predictive_results.csv y .json con todas las 10,584 filas

Requisito:
  Correr primero el backtest con todos los USE_* = True y umbral bajo
  (ej. 0.3) para capturar todos los pred_* en el trade_history.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import sys
import time
from collections import defaultdict
from itertools import combinations as iter_combinations
from pathlib   import Path
from typing    import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL

from actors.price_feed   import SQLiteFeed
from actors.wallet       import JSONWallet, MemoryWallet, TradeRecord
from actors.order_book   import SimulatedOrderBook, OrderSide
from actors.clock        import LocalClock
from risk.risk_manager   import RiskManager, RiskConfig
from state.state_manager import MemoryStateManager, Checkpoint
from strategies.predictive_candles import (
    PredictiveCandlesStrategy,
    AUC_BOT, AUC_TOP, DIR_BOT, DIR_TOP,
    ALL_BOT_PREDICTORS, ALL_TOP_PREDICTORS,
    norm_weights,
)
from strategies.base_strategy import SignalSide
from support.logger import get_logger

log = get_logger("backtest_predictive_candles")


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — Editar aquí antes de cada ejecución
# ══════════════════════════════════════════════════════════════════════════════

VENTANA    = 10     # velas hacia atrás para calcular factores  [10, 16]
UMBRAL_BOT = 0.8   # score mínimo para señal BUY               (0, 1]
UMBRAL_TOP = 0.7   # score mínimo para señal SELL              (0, 1]
COOLDOWN_BOT = 96    # velas mínimas entre señales BUY  (0 = desactivado)
COOLDOWN_TOP = 96    # velas mínimas entre señales SELL (0 = desactivado)

# ── Predictores activos para BOTTOM (señal BUY) ───────────────────────────────
USE_BOT_CLOSE_POSITION = True   # AUC=0.854 ▼
USE_BOT_BB_POSITION    = True   # AUC=0.833 ▼
USE_BOT_RECOVERY_PCT   = True   # AUC=0.823 ▼

# ── Predictores activos para TOP (señal SELL) ─────────────────────────────────
USE_TOP_CLOSE_POSITION = True   # AUC=0.839 ▲
USE_TOP_DRAWDOWN_PCT   = True   # AUC=0.826 ▼
USE_TOP_BB_POSITION    = True   # AUC=0.821 ▲

# ── Salida ────────────────────────────────────────────────────────────────────
RESULTS_JSON = CL.RESULTS_JSON

# ── Parámetros del grid ───────────────────────────────────────────────────────
# Umbrales a barrer (aplica tanto a BOT como a TOP por separado)
GRID_THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
# Cooldowns a barrer (0 = desactivado, resto en velas)
GRID_COOLDOWNS  = [0, 12, 24, 48, 72, 96]
GRID_CSV        = "grid_predictive_results.csv"
GRID_JSON_OUT   = "grid_predictive_results.json"

_ABBREV = {
    "close_position": "cp",
    "bb_position":    "bb",
    "recovery_pct":   "rc",
    "drawdown_pct":   "dw",
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER: enriquecer trade_log con valores de predictores y scores
# ══════════════════════════════════════════════════════════════════════════════

def _enrich(wallet: JSONWallet, strategy: PredictiveCandlesStrategy) -> None:
    """Agrega pred_* y scores al último trade del log."""
    entries = wallet.get_trade_log()
    if not entries:
        return
    last = entries[-1]
    last["score_bot"]           = strategy.last_score_bot
    last["score_top"]           = strategy.last_score_top
    last["pred_close_position"] = strategy.last_pred_values.get("close_position")
    last["pred_bb_position"]    = strategy.last_pred_values.get("bb_position")
    last["pred_recovery_pct"]   = strategy.last_pred_values.get("recovery_pct")
    last["pred_drawdown_pct"]   = strategy.last_pred_values.get("drawdown_pct")


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    t_start = time.time()

    bot_on = [k for k, v in {
        "close_pos": USE_BOT_CLOSE_POSITION,
        "bb_pos":    USE_BOT_BB_POSITION,
        "recovery":  USE_BOT_RECOVERY_PCT,
    }.items() if v]
    top_on = [k for k, v in {
        "close_pos": USE_TOP_CLOSE_POSITION,
        "drawdown":  USE_TOP_DRAWDOWN_PCT,
        "bb_pos":    USE_TOP_BB_POSITION,
    }.items() if v]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║      BACKTEST PREDICTIVE CANDLES — BTC/USDT             ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Rango          : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital        : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
    print(f"  Max posiciones : {CL.MAX_POSICIONES}")
    print(f"  Comisión       : {CL.COMMISSION_PCT}%")
    print(f"  Ventana        : {VENTANA} velas")
    print(f"  Umbral BOT/TOP : {UMBRAL_BOT} / {UMBRAL_TOP}")
    print(f"  Cooldown BOT   : {COOLDOWN_BOT} velas" if COOLDOWN_BOT else "  Cooldown BOT   : desactivado")
    print(f"  Cooldown TOP   : {COOLDOWN_TOP} velas" if COOLDOWN_TOP else "  Cooldown TOP   : desactivado")
    print(f"  Pred. BOT      : {', '.join(bot_on) if bot_on else 'NINGUNO'}")
    print(f"  Pred. TOP      : {', '.join(top_on) if top_on else 'NINGUNO'}")
    print(f"  Output JSON    : {RESULTS_JSON}")
    print("─" * 60)

    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    clock  = LocalClock(feed=feed, start=CL.FECHA_INICIO, end=CL.FECHA_FIN,
                        symbol=CL.SYMBOL)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = RESULTS_JSON,
    )
    ob    = SimulatedOrderBook(
        commission_pct = CL.COMMISSION_PCT,
        max_posiciones = CL.MAX_POSICIONES,
    )
    risk  = RiskManager(config=RiskConfig.permissive(),
                        usdt_inicial=CL.SALDO_USDT_INICIAL)
    state = MemoryStateManager()

    strategy = PredictiveCandlesStrategy(
        ventana                = VENTANA,
        umbral_bot             = UMBRAL_BOT,
        umbral_top             = UMBRAL_TOP,
        use_bot_close_position = USE_BOT_CLOSE_POSITION,
        use_bot_bb_position    = USE_BOT_BB_POSITION,
        use_bot_recovery_pct   = USE_BOT_RECOVERY_PCT,
        use_top_close_position = USE_TOP_CLOSE_POSITION,
        use_top_drawdown_pct   = USE_TOP_DRAWDOWN_PCT,
        use_top_bb_position    = USE_TOP_BB_POSITION,
        cooldown_bot           = COOLDOWN_BOT,
        cooldown_top           = COOLDOWN_TOP,
    )
    strategy.on_start(wallet)

    n_compras = n_ventas = n_ignorados = 0
    ign_motivos: Dict[str, int] = {}
    precio_min_comprado = float("inf")
    precio_max_vendido  = float("-inf")
    last_candle         = None

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
                precio_min_comprado = min(precio_min_comprado, signal.price)
            else:
                n_ventas += 1
                precio_max_vendido = max(precio_max_vendido, signal.price)
        else:
            n_ignorados += 1
            motivo = order.reject_reason or "desconocido"
            ign_motivos[motivo] = ign_motivos.get(motivo, 0) + 1

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(
            wallet, candle.ts, candle.close,
            metadata={"estrategia": strategy.name},
        ))

    print(f"OK  ({clock.total_candles:,} velas)")

    strategy.on_stop(wallet)

    if last_candle is None:
        print("✗ No se encontraron velas en el rango indicado.")
        return

    precio_final   = last_candle.close
    port_final     = wallet.portfolio_value(precio_final)
    pnl_pct        = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100
    first_candles  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_inicial = first_candles[0].close if first_candles else precio_final
    bh_pnl         = (precio_final / precio_inicial - 1) * 100
    all_candles    = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl = min(c.low  for c in all_candles) if all_candles else 0
    ath = max(c.high for c in all_candles) if all_candles else 0

    summary = {
        "estrategia":               strategy.name,
        "fecha_inicio":             CL.FECHA_INICIO,
        "fecha_fin":                CL.FECHA_FIN,
        "saldo_inicial_usdt":       CL.SALDO_USDT_INICIAL,
        "usdt_balance_final":       round(wallet.get_usdt_balance(), 8),
        "btc_balance_final":        round(wallet.get_btc_balance(), 10),
        "btc_acumulado_total":      round(wallet.get_btc_acumulado(), 10),
        "btc_en_posiciones_final":  round(wallet.btc_en_posiciones(), 10),
        "precio_promedio_final":    round(wallet.precio_promedio_posiciones(), 8),
        "portfolio_value_final":    round(port_final, 4),
        "pnl_pct":                  round(pnl_pct, 4),
        "buy_hold_pnl_pct":         round(bh_pnl, 4),
        "alpha_vs_bh":              round(pnl_pct - bh_pnl, 4),
        "precio_min_comprado":      round(precio_min_comprado, 4) if n_compras else None,
        "precio_max_vendido":       round(precio_max_vendido,  4) if n_ventas  else None,
        "atl_final":                round(atl, 4),
        "ath_proyectado_final":     round(ath, 4),
        "total_trades_ejecutados":  n_compras + n_ventas,
        "total_compras":            n_compras,
        "total_ventas":             n_ventas,
        "total_ignorados":          n_ignorados,
        "ordenes_canceladas":       0,
        "ignorados_por_motivo":     ign_motivos,
        "positions_count_final":    wallet.positions_count,
        "usdt_reserva_aplicada":    0.0,
        "umbral_filtro":            UMBRAL_BOT,
        "parametros":               strategy.describe(),
    }

    wallet.flush(summary)

    elapsed = time.time() - t_start
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
    print(f"\n✓ Resultado guardado en: {RESULTS_JSON}")
    print(f"\n  Flujo recomendado para --grid:")
    print(f"  1. Configurar todos USE_* = True y umbrales bajos (0.3)")
    print(f"  2. python backtest_predictive_candles.py")
    print(f"  3. python backtest_predictive_candles.py --grid")


# ══════════════════════════════════════════════════════════════════════════════
# MODO --grid: funciones auxiliares
# ══════════════════════════════════════════════════════════════════════════════

def _pct_rank_fast(value: float, sorted_arr: List[float]) -> float:
    """Percentil con bisect. O(log n)."""
    if not sorted_arr:
        return 0.5
    return bisect.bisect_left(sorted_arr, value) / len(sorted_arr)


def _grid_score(
    active:     List[str],
    weights:    Dict[str, float],
    dirs:       Dict[str, str],
    pv:         Dict[str, Optional[float]],
    sorted_rec: List[float],
    sorted_dra: List[float],
) -> float:
    """Score ponderado para una combinación de predictores (versión grid, O(log n))."""
    if not active:
        return 0.0
    score = w_total = 0.0
    for pred in active:
        raw = pv.get(pred)
        if raw is None:
            continue
        w = weights.get(pred, 0.0)
        if pred in ("close_position", "bb_position"):
            norm = raw
        elif pred == "recovery_pct":
            norm = _pct_rank_fast(raw, sorted_rec) if sorted_rec else 0.5
        else:
            norm = _pct_rank_fast(raw, sorted_dra) if sorted_dra else 0.5
        score   += w * ((1.0 - norm) if dirs[pred] == "LOW" else norm)
        w_total += w
    if w_total <= 0:
        return 0.0
    return score / w_total if abs(w_total - 1.0) > 1e-9 else score


def _simulate_combo(
    trades:          List[dict],
    bot_combo:       List[str],
    top_combo:       List[str],
    thr_bot:         float,
    thr_top:         float,
    cooldown:        int,
    rec_windows:     List[List[float]],   # ventana rolling causal por trade
    dra_windows:     List[List[float]],   # ventana rolling causal por trade
    usdt_ini:        float,
    max_pos:         int,
    commission:      float,
) -> Dict:
    """
    Re-simula la wallet con una combinación específica de predictores,
    umbrales y cooldown.

    rec_windows[i] / dra_windows[i]: lista ORDENADA de los últimos n_norm
    valores de recovery_pct / drawdown_pct vistos hasta el trade i inclusive.
    Estas ventanas replican exactamente el historial rolling de la estrategia,
    eliminando el look-ahead bias que existía en la versión anterior que usaba
    la distribución global del período completo.

    cooldown se aplica simétricamente a BOT y TOP.
    """
    w_bot = norm_weights(bot_combo, AUC_BOT) if bot_combo else {}
    w_top = norm_weights(top_combo, AUC_TOP) if top_combo else {}

    wallet = MemoryWallet(usdt_ini, max_pos)
    n_buy = n_sell = n_ign = 0
    ganancias: List[float] = []

    _NEG_INF   = -(10 ** 9)
    last_bot_i = _NEG_INF
    last_top_i = _NEG_INF

    for trade_idx, t in enumerate(trades):
        side  = t.get("type", "")
        price = t.get("price", 0.0)
        if price <= 0 or side not in ("BUY", "SELL"):
            continue

        pv = {
            "close_position": t.get("pred_close_position"),
            "bb_position":    t.get("pred_bb_position"),
            "recovery_pct":   t.get("pred_recovery_pct"),
            "drawdown_pct":   t.get("pred_drawdown_pct"),
        }

        # Usar la ventana causal para este trade_idx
        s_rec = rec_windows[trade_idx]
        s_dra = dra_windows[trade_idx]

        if side == "BUY":
            if not bot_combo:
                continue
            if cooldown > 0 and (trade_idx - last_bot_i) < cooldown:
                continue
            sb = _grid_score(bot_combo, w_bot, DIR_BOT, pv, s_rec, s_dra)
            if sb < thr_bot:
                continue
            # Guardias de wallet
            if wallet.positions_count >= max_pos:
                n_ign += 1
                continue
            slot = wallet.get_slot_usdt()
            if slot > wallet.get_usdt_balance() + 1e-9:
                n_ign += 1
                continue
            comm = slot * commission / 100.0
            btc  = (slot - comm) / price
            wallet.update(TradeRecord(
                ts=0, side="BUY", price=price,
                usdt_spent=slot, btc_bought=btc, commission=comm,
            ))
            last_bot_i = trade_idx
            n_buy += 1

        elif side == "SELL":
            if not top_combo:
                continue
            if cooldown > 0 and (trade_idx - last_top_i) < cooldown:
                continue
            st = _grid_score(top_combo, w_top, DIR_TOP, pv, s_rec, s_dra)
            if st < thr_top:
                continue
            # Guardias de wallet
            if wallet.positions_count == 0:
                n_ign += 1
                continue
            bpv = wallet.get_btc_por_venta()
            if bpv <= 0:
                n_ign += 1
                continue
            usdt_bruto = bpv * price
            comm       = usdt_bruto * commission / 100.0
            usdt_neto  = usdt_bruto - comm
            ganancia   = usdt_neto - wallet.get_slot_usdt()
            wallet.update(TradeRecord(
                ts=0, side="SELL", price=price,
                btc_sold=bpv, usdt_received=usdt_neto,
                commission=comm, ganancia_usdt=ganancia,
            ))
            ganancias.append(ganancia)
            last_top_i = trade_idx
            n_sell += 1

    last_price = trades[-1].get("price", 0.0) if trades else 0.0
    port  = wallet.portfolio_value(last_price)
    pnl   = (port / usdt_ini - 1) * 100.0
    wr    = (sum(1 for g in ganancias if g > 0) / len(ganancias) * 100.0
             if ganancias else 0.0)

    return {
        "pnl_pct":  round(pnl, 2),
        "portfolio": round(port, 2),
        "n_buy":    n_buy,
        "n_sell":   n_sell,
        "n_trades": n_buy + n_sell,
        "n_ign":    n_ign,
        "win_rate": round(wr, 1),
    }


def _nonempty_subsets(lst: List[str]) -> List[List[str]]:
    """Todos los subconjuntos no vacíos."""
    result = []
    for r in range(1, len(lst) + 1):
        for combo in iter_combinations(lst, r):
            result.append(list(combo))
    return result


def _abbrev(combo: List[str]) -> str:
    return "+".join(_ABBREV.get(p, p) for p in combo)


# ══════════════════════════════════════════════════════════════════════════════
# MODO --grid: tabla de análisis por predictor
# ══════════════════════════════════════════════════════════════════════════════

def _build_predictor_table(results: List[dict], bh_pnl: float) -> None:
    """
    Muestra dos tablas:
      1. Ranking de combinaciones BOT: para cada subconjunto de predictores
         BOT, cuál es el mejor PnL alcanzable (optimizando top, thr, cooldown)
      2. Ídem para TOP.

    Columnas: combo | mejor PnL | alpha vs B&H | thr_bot/top | cooldown | B/S | WR%
    """
    sep  = "─" * 82
    sep2 = "═" * 82

    # ── Tabla BOT: agrupar por bot_combo, encontrar mejor resultado ───────────
    by_bot: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_bot[r["bot_combo"]].append(r)

    best_by_bot = []
    for bot_c, rows in by_bot.items():
        best = max(rows, key=lambda x: x["pnl_pct"])
        best_by_bot.append({
            "combo":    bot_c,
            "abbrev":   _abbrev(bot_c.split("+")),
            "n_pred":   len(bot_c.split("+")),
            "is_solo":  len(bot_c.split("+")) == 1,
            "pnl":      best["pnl_pct"],
            "alpha_bh": best["alpha_vs_bh"],
            "thr_b":    best["umbral_bot"],
            "thr_t":    best["umbral_top"],
            "cooldown": best["cooldown"],
            "n_buy":    best["n_buy"],
            "n_sell":   best["n_sell"],
            "win_rate": best["win_rate"],
            "best_top": _abbrev(best["top_combo"].split("+")),
        })
    best_by_bot.sort(key=lambda x: -x["pnl"])

    print(f"\n{sep2}")
    print("  RANKING BOT — mejor PnL por combinación de predictores de BOTTOM")
    print(f"  (optimizando top_combo, umbrales y cooldown)")
    print(sep2)
    print(f"  {'#':>2}  {'BOT':>14}  {'mejor_top':>14}  "
          f"{'thr_b':>6} {'thr_t':>6} {'cd':>4}  "
          f"{'PnL%':>8} {'α_BH':>8}  {'B/S':>9}  {'WR%':>5}")
    print(sep)
    for i, r in enumerate(best_by_bot, 1):
        solo_tag = " *" if r["is_solo"] else "  "
        pnl_s = f"{'+' if r['pnl']>=0 else ''}{r['pnl']:.2f}%"
        abh_s = f"{'+' if r['alpha_bh']>=0 else ''}{r['alpha_bh']:.2f}%"
        cd_s  = str(int(r["cooldown"])) if r["cooldown"] else "off"
        bs_s  = f"{r['n_buy']}B/{r['n_sell']}S"
        print(f"  {i:>2}.{solo_tag}{r['abbrev']:>14}  {r['best_top']:>14}  "
              f"{r['thr_b']:>6.2f} {r['thr_t']:>6.2f} {cd_s:>4}  "
              f"{pnl_s:>9} {abh_s:>9}  {bs_s:>9}  {r['win_rate']:>5.1f}%")
    print(sep)
    print("  * = predictor individual (sin combinación)")
    print(f"  B&H referencia: {bh_pnl:+.2f}%")

    # ── Tabla TOP: ídem ───────────────────────────────────────────────────────
    by_top: Dict[str, List[dict]] = defaultdict(list)
    for r in results:
        by_top[r["top_combo"]].append(r)

    best_by_top = []
    for top_c, rows in by_top.items():
        best = max(rows, key=lambda x: x["pnl_pct"])
        best_by_top.append({
            "combo":    top_c,
            "abbrev":   _abbrev(top_c.split("+")),
            "n_pred":   len(top_c.split("+")),
            "is_solo":  len(top_c.split("+")) == 1,
            "pnl":      best["pnl_pct"],
            "alpha_bh": best["alpha_vs_bh"],
            "thr_b":    best["umbral_bot"],
            "thr_t":    best["umbral_top"],
            "cooldown": best["cooldown"],
            "n_buy":    best["n_buy"],
            "n_sell":   best["n_sell"],
            "win_rate": best["win_rate"],
            "best_bot": _abbrev(best["bot_combo"].split("+")),
        })
    best_by_top.sort(key=lambda x: -x["pnl"])

    print(f"\n{sep2}")
    print("  RANKING TOP — mejor PnL por combinación de predictores de TOP")
    print(f"  (optimizando bot_combo, umbrales y cooldown)")
    print(sep2)
    print(f"  {'#':>2}  {'TOP':>14}  {'mejor_bot':>14}  "
          f"{'thr_b':>6} {'thr_t':>6} {'cd':>4}  "
          f"{'PnL%':>8} {'α_BH':>8}  {'B/S':>9}  {'WR%':>5}")
    print(sep)
    for i, r in enumerate(best_by_top, 1):
        solo_tag = " *" if r["is_solo"] else "  "
        pnl_s = f"{'+' if r['pnl']>=0 else ''}{r['pnl']:.2f}%"
        abh_s = f"{'+' if r['alpha_bh']>=0 else ''}{r['alpha_bh']:.2f}%"
        cd_s  = str(int(r["cooldown"])) if r["cooldown"] else "off"
        bs_s  = f"{r['n_buy']}B/{r['n_sell']}S"
        print(f"  {i:>2}.{solo_tag}{r['abbrev']:>14}  {r['best_bot']:>14}  "
              f"{r['thr_b']:>6.2f} {r['thr_t']:>6.2f} {cd_s:>4}  "
              f"{pnl_s:>9} {abh_s:>9}  {bs_s:>9}  {r['win_rate']:>5.1f}%")
    print(sep)
    print("  * = predictor individual (sin combinación)")

    # ── Tabla de predictores individuales únicos ──────────────────────────────
    print(f"\n{sep2}")
    print("  PREDICTORES INDIVIDUALES — comparación directa (solo predictor único activo)")
    print(sep2)
    print(f"  {'tipo':>4}  {'predictor':>16}  {'mejor_contraparte':>18}  "
          f"{'thr_b':>6} {'thr_t':>6} {'cd':>4}  "
          f"{'PnL%':>8} {'α_BH':>8}  {'WR%':>5}")
    print(sep)

    # Solos de BOT
    for r in best_by_bot:
        if not r["is_solo"]:
            continue
        pnl_s = f"{'+' if r['pnl']>=0 else ''}{r['pnl']:.2f}%"
        abh_s = f"{'+' if r['alpha_bh']>=0 else ''}{r['alpha_bh']:.2f}%"
        cd_s  = str(int(r["cooldown"])) if r["cooldown"] else "off"
        print(f"  {'BOT':>4}  {r['abbrev']:>16}  {r['best_top']:>18}  "
              f"{r['thr_b']:>6.2f} {r['thr_t']:>6.2f} {cd_s:>4}  "
              f"{pnl_s:>9} {abh_s:>9}  {r['win_rate']:>5.1f}%")

    # Solos de TOP
    for r in best_by_top:
        if not r["is_solo"]:
            continue
        pnl_s = f"{'+' if r['pnl']>=0 else ''}{r['pnl']:.2f}%"
        abh_s = f"{'+' if r['alpha_bh']>=0 else ''}{r['alpha_bh']:.2f}%"
        cd_s  = str(int(r["cooldown"])) if r["cooldown"] else "off"
        print(f"  {'TOP':>4}  {r['abbrev']:>16}  {r['best_bot']:>18}  "
              f"{r['thr_b']:>6.2f} {r['thr_t']:>6.2f} {cd_s:>4}  "
              f"{pnl_s:>9} {abh_s:>9}  {r['win_rate']:>5.1f}%")

    print(sep)
    print("  Abreviaciones: cp=close_position  bb=bb_position  "
          "rc=recovery_pct  dw=drawdown_pct")


# ══════════════════════════════════════════════════════════════════════════════
# MODO --grid: análisis principal
# ══════════════════════════════════════════════════════════════════════════════

def grid_analysis(json_path: str) -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      GRID — Análisis de Combinaciones de Predictores    ║")
    print("╚══════════════════════════════════════════════════════════╝")

    json_file = Path(json_path)
    if not json_file.exists():
        print(f"\n✗ No se encontró: {json_path}")
        print("  Paso 1: todos los USE_* = True, umbrales bajos (0.3)")
        print("  Paso 2: python backtest_predictive_candles.py")
        print("  Paso 3: python backtest_predictive_candles.py --grid")
        return

    with open(json_file, encoding="utf-8") as f:
        data = json.load(f)

    trades_raw  = data.get("trade_history", [])
    summary_in  = data.get("summary", {})
    orig_pnl    = summary_in.get("pnl_pct", 0.0)
    bh_pnl      = summary_in.get("buy_hold_pnl_pct", 0.0)

    print(f"  JSON           : {json_path}")
    print(f"  Período        : {summary_in.get('fecha_inicio')} → {summary_in.get('fecha_fin')}")
    print(f"  PnL original   : {orig_pnl:+.2f}%   |   B&H: {bh_pnl:+.2f}%")
    print(f"  Trades totales : {len(trades_raw):,}")

    # Verificar campos pred_*
    required = ["pred_close_position", "pred_bb_position",
                "pred_recovery_pct",   "pred_drawdown_pct"]
    action_trades = [t for t in trades_raw if t.get("type") in ("BUY", "SELL")]
    missing = [f for f in required
               if not any(t.get(f) is not None for t in action_trades)]

    if missing:
        print(f"\n✗ Campos faltantes en trade_history: {missing}")
        print("\n  Para habilitar el --grid:")
        print("  1. Todos los USE_* = True y umbrales bajos (ej. 0.3)")
        print("  2. python backtest_predictive_candles.py")
        print("  3. python backtest_predictive_candles.py --grid")
        return

    trades = [t for t in action_trades
              if t.get("pred_close_position") is not None]
    n_sin_pred = len(action_trades) - len(trades)
    if n_sin_pred:
        print(f"  ⚠ {n_sin_pred} trades sin pred_values — se excluyen del grid")
    print(f"  Trades con pred_values: {len(trades):,}  ✓")

    # ── Pre-computar ventanas rolling causales por trade ──────────────────────
    # Replica exactamente el historial h_rec / h_dra de la estrategia real:
    # rec_windows[i] = lista ORDENADA de los últimos N_NORM valores de
    # recovery_pct vistos hasta el trade i inclusive.
    # Esto elimina el look-ahead bias que causaba la discrepancia entre
    # el grid (distribución global del período completo) y el backtest real
    # (ventana rolling de n_norm=200 observaciones).
    N_NORM = 200
    print(f"  Precomputando ventanas causales (n_norm={N_NORM})...", end=" ", flush=True)
    rec_windows: List[List[float]] = []
    dra_windows: List[List[float]] = []
    h_rec_roll: List[float] = []   # ventana FIFO sin ordenar
    h_dra_roll: List[float] = []

    for t_roll in trades:
        rv = t_roll.get("pred_recovery_pct")
        dv = t_roll.get("pred_drawdown_pct")
        if rv is not None:
            h_rec_roll.append(rv)
            if len(h_rec_roll) > N_NORM:
                h_rec_roll.pop(0)
        if dv is not None:
            h_dra_roll.append(dv)
            if len(h_dra_roll) > N_NORM:
                h_dra_roll.pop(0)
        # Guardar copia ordenada para bisect rápido en _grid_score
        rec_windows.append(sorted(h_rec_roll))
        dra_windows.append(sorted(h_dra_roll))

    print(f"OK ({len(rec_windows):,} ventanas)")

    usdt_ini   = summary_in.get("saldo_inicial_usdt", CL.SALDO_USDT_INICIAL)
    max_pos    = (summary_in.get("parametros", {})
                  .get("max_posiciones", CL.MAX_POSICIONES))
    commission = CL.COMMISSION_PCT

    bot_combos  = _nonempty_subsets(ALL_BOT_PREDICTORS)   # 7 subconjuntos
    top_combos  = _nonempty_subsets(ALL_TOP_PREDICTORS)   # 7 subconjuntos
    thresholds  = GRID_THRESHOLDS
    cooldowns   = GRID_COOLDOWNS
    total_combos = len(bot_combos) * len(top_combos) * len(thresholds) ** 2 * len(cooldowns)

    print(f"\n  Combinaciones  : {len(bot_combos)} BOT × {len(top_combos)} TOP"
          f" × {len(thresholds)}² umbrales × {len(cooldowns)} cooldowns"
          f" = {total_combos:,}")
    print(f"  Umbrales       : {thresholds}")
    print(f"  Cooldowns      : {cooldowns} (velas; 0=off)")
    print(f"\n  Ejecutando simulaciones...")

    t0      = time.time()
    results = []
    done    = 0

    for bc in bot_combos:
        for tc in top_combos:
            for thr_b in thresholds:
                for thr_t in thresholds:
                    for cd in cooldowns:
                        metrics = _simulate_combo(
                            trades, bc, tc, thr_b, thr_t, cd,
                            rec_windows, dra_windows,
                            usdt_ini, max_pos, commission,
                        )
                        results.append({
                            "bot_combo": "+".join(bc),
                            "top_combo": "+".join(tc),
                            "umbral_bot": thr_b,
                            "umbral_top": thr_t,
                            "cooldown":   cd,
                            **metrics,
                            "alpha_vs_bh":   round(metrics["pnl_pct"] - bh_pnl,   2),
                            "alpha_vs_orig": round(metrics["pnl_pct"] - orig_pnl, 2),
                        })
                        done += 1

            elapsed_so_far = time.time() - t0
            pct = done / total_combos * 100
            print(f"  {pct:5.1f}%  ({done:,}/{total_combos:,})  "
                  f"{elapsed_so_far:.1f}s", end="\r", flush=True)

    elapsed = time.time() - t0
    print(f"  100.0%  ({total_combos:,}/{total_combos:,})  {elapsed:.1f}s  ✓")

    results.sort(key=lambda x: -x["pnl_pct"])

    # ── Top-15 general ────────────────────────────────────────────────────────
    _print_top15(results[:15], orig_pnl, bh_pnl)

    # ── Tabla de ranking por predictor ────────────────────────────────────────
    _build_predictor_table(results, bh_pnl)

    # ── Guardar outputs ───────────────────────────────────────────────────────
    print("\n[Guardando resultados...]")
    _save_grid_csv(results, GRID_CSV)
    _save_grid_json(results, GRID_JSON_OUT, summary_in, elapsed)
    print(f"\n✓ Grid completado en {elapsed:.1f}s")
    print(f"  {GRID_CSV}  |  {GRID_JSON_OUT}")


def _print_top15(results: List[dict], orig_pnl: float, bh_pnl: float) -> None:
    sep  = "─" * 96
    sep2 = "═" * 96
    print(f"\n{sep2}")
    print("  TOP-15 COMBINACIONES — ordenadas por PnL")
    print(sep2)
    print(
        f"  {'#':>2}  {'BOT':>8} {'TOP':>12}  "
        f"{'thr_b':>6} {'thr_t':>6} {'cd':>4}  "
        f"{'PnL%':>8} {'α_BH':>8} {'α_orig':>8}  "
        f"{'B/S':>9}  {'WR%':>5}"
    )
    print(sep)
    for i, r in enumerate(results, 1):
        pnl_s  = f"{'+' if r['pnl_pct']>=0 else ''}{r['pnl_pct']:.2f}%"
        abh_s  = f"{'+' if r['alpha_vs_bh']>=0 else ''}{r['alpha_vs_bh']:.2f}%"
        aor_s  = f"{'+' if r['alpha_vs_orig']>=0 else ''}{r['alpha_vs_orig']:.2f}%"
        cd_s   = str(int(r["cooldown"])) if r["cooldown"] else "off"
        bc_s   = _abbrev(r["bot_combo"].split("+"))
        tc_s   = _abbrev(r["top_combo"].split("+"))
        bs_s   = f"{r['n_buy']}B/{r['n_sell']}S"
        print(
            f"  {i:>2}.  {bc_s:>8} {tc_s:>12}  "
            f"{r['umbral_bot']:>6.2f} {r['umbral_top']:>6.2f} {cd_s:>4}  "
            f"{pnl_s:>9} {abh_s:>9} {aor_s:>9}  "
            f"{bs_s:>9}  {r['win_rate']:>5.1f}%"
        )
    print(sep)
    print(f"  Original → PnL: {orig_pnl:+.2f}%   B&H: {bh_pnl:+.2f}%")
    print(sep2)
    print("  Abreviaciones: cp=close_position  bb=bb_position  "
          "rc=recovery_pct  dw=drawdown_pct  cd=cooldown (velas)")


def _save_grid_csv(results: List[dict], path: str) -> None:
    if not results:
        return
    fieldnames = list(results[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)
    print(f"  ✓ CSV  → {path}  ({len(results):,} filas)")


def _save_grid_json(
    results: List[dict], path: str,
    orig_summary: dict, elapsed: float,
) -> None:
    output = {
        "meta": {
            "fecha_inicio":    orig_summary.get("fecha_inicio"),
            "fecha_fin":       orig_summary.get("fecha_fin"),
            "pnl_original":    orig_summary.get("pnl_pct"),
            "bh_original":     orig_summary.get("buy_hold_pnl_pct"),
            "grid_thresholds": GRID_THRESHOLDS,
            "grid_cooldowns":  GRID_COOLDOWNS,
            "n_combinaciones": len(results),
            "elapsed_s":       round(elapsed, 1),
        },
        "top_10": results[:10],
        "all":    results,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"  ✓ JSON → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest PredictiveCandles — Score ponderado por AUC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python backtest_predictive_candles.py
  python backtest_predictive_candles.py --grid
  python backtest_predictive_candles.py --grid --json otro_resultado.json

Flujo recomendado para --grid:
  1. Todos los USE_* = True, UMBRAL_BOT/TOP = 0.3, COOLDOWN_BOT/TOP = 0
  2. python backtest_predictive_candles.py        (genera JSON con pred_*)
  3. python backtest_predictive_candles.py --grid (10,584 combinaciones)
  4. Configurar la combo óptima encontrada y re-correr el backtest definitivo
""",
    )
    parser.add_argument(
        "--grid", action="store_true",
        help="Analiza combinaciones sobre el trade_history existente",
    )
    parser.add_argument(
        "--json", default=RESULTS_JSON, metavar="PATH",
        help=f"JSON de resultados para el modo --grid (default: {RESULTS_JSON})",
    )
    args = parser.parse_args()

    if args.grid:
        grid_analysis(args.json)
    else:
        main()