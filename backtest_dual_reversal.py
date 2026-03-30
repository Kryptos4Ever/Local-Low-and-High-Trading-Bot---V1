"""
backtest_dual_reversal.py — Runner de la Estrategia Dual Reversal
══════════════════════════════════════════════════════════════════
Uso:
    python backtest_dual_reversal.py           # run con parámetros default
    python backtest_dual_reversal.py --grid    # grid search completo
    python backtest_dual_reversal.py --grid --top 20   # mostrar top N

El grid search varía tanto los parámetros de los indicadores (períodos RSI,
MAs, ventanas) como los umbrales de ambas capas, cubriendo el espacio de
configuración completo de la estrategia.

Configuración del entorno: config_local.py
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL

from actors.price_feed           import SQLiteFeed
from actors.wallet               import JSONWallet, MemoryWallet, TradeRecord
from actors.order_book           import SimulatedOrderBook, OrderSide
from actors.clock                import LocalClock
from risk.risk_manager           import RiskManager, RiskConfig
from state.state_manager         import MemoryStateManager, Checkpoint
from strategies.dual_reversal    import DualReversalStrategy
from strategies.base_strategy    import SignalSide
from support.logger              import get_logger

log = get_logger("backtest_dual_reversal")


# ════════════════════════════════════════════════════════════════════
# PARÁMETROS DEFAULT
# ════════════════════════════════════════════════════════════════════

# Indicadores
RSI_PERIOD      = 20
MA_SHORT        = 25
MA_LONG         = 50
WINDOW          = 24     # historia para contexto
LAST_N          = 12      # ventana del disparador

# Contexto — umbrales
CTX_RSI_BUY     = 38.4   # p75 BOTTOMs = 38.4
CTX_RSI_SELL    = 61.6   # p25 TOPs    = 54.4
CTX_MA20_BUY    = -7.0   # p50 BOTTOMs price_vs_ma20 = -2.47%
CTX_MA20_SELL   =  7.0   # p50 TOPs    price_vs_ma20 = +1.85%
CTX_MIN_PTS     = 5

# Disparador — umbrales
TRIG_RSI_SLOPE  = 0.07   # p25 |rsi_slope| BOTTOMs = 0.026
TRIG_WICK       = 0.46   # p50 lower_wick BOTTOMs  = 0.292
TRIG_MIN_PTS    = 5


# ════════════════════════════════════════════════════════════════════
# RUNNER ÚNICO
# ════════════════════════════════════════════════════════════════════

def run_backtest(
    # Indicadores
    rsi_period:      int   = RSI_PERIOD,
    ma_short:        int   = MA_SHORT,
    ma_long:         int   = MA_LONG,
    window:          int   = WINDOW,
    last_n:          int   = LAST_N,
    # Contexto
    ctx_rsi_buy:     float = CTX_RSI_BUY,
    ctx_rsi_sell:    float = CTX_RSI_SELL,
    ctx_ma20_buy:    float = CTX_MA20_BUY,
    ctx_ma20_sell:   float = CTX_MA20_SELL,
    ctx_min_pts:     int   = CTX_MIN_PTS,
    # Disparador
    trig_rsi_slope:  float = TRIG_RSI_SLOPE,
    trig_wick:       float = TRIG_WICK,
    trig_min_pts:    int   = TRIG_MIN_PTS,
    # Output
    results_json:    str   = None,
    silent:          bool  = False,
) -> dict:

    t_start   = time.time()
    json_path = results_json or CL.RESULTS_JSON

    warmup = max(ma_long + last_n + 4, rsi_period + last_n + 4, window + last_n + 4) + 4

    if not silent:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║     BACKTEST DUAL REVERSAL — 2 Capas BTC/USDT           ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Rango          : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
        print(f"  Capital        : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
        print(f"  Comisión       : {CL.COMMISSION_PCT}%")
        print(f"  Indicadores    : RSI({rsi_period})  MA{ma_short}/{ma_long}  W={window}  N={last_n}")
        print(f"  Contexto       : RSI<{ctx_rsi_buy}/>{ ctx_rsi_sell}  MA20<{ctx_ma20_buy}%/>{ctx_ma20_sell}%  min={ctx_min_pts}pts")
        print(f"  Disparador     : rsi_slope>{trig_rsi_slope}  wick>{trig_wick}  min={trig_min_pts}pts")
        print(f"  Output JSON    : {json_path}")
        print("─" * 60)

    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = json_path,
    )
    ob     = SimulatedOrderBook(
        commission_pct = CL.COMMISSION_PCT,
        max_posiciones = CL.MAX_POSICIONES,
    )
    risk   = RiskManager(config=RiskConfig.permissive(),
                         usdt_inicial=CL.SALDO_USDT_INICIAL)
    state  = MemoryStateManager()

    strategy = DualReversalStrategy(
        rsi_period     = rsi_period,
        ma_short       = ma_short,
        ma_long        = ma_long,
        window         = window,
        last_n         = last_n,
        ctx_rsi_buy    = ctx_rsi_buy,
        ctx_rsi_sell   = ctx_rsi_sell,
        ctx_ma20_buy   = ctx_ma20_buy,
        ctx_ma20_sell  = ctx_ma20_sell,
        ctx_min_pts    = ctx_min_pts,
        trig_rsi_slope = trig_rsi_slope,
        trig_wick      = trig_wick,
        trig_min_pts   = trig_min_pts,
        warmup         = warmup,
    )
    strategy.on_start(wallet)

    clock = LocalClock(
        feed=feed, start=CL.FECHA_INICIO, end=CL.FECHA_FIN, symbol=CL.SYMBOL
    )

    n_compras = n_ventas = n_ignorados = 0
    ign_motivos: dict[str, int] = {}
    precio_min_comprado = float("inf")
    precio_max_vendido  = float("-inf")
    last_candle = None

    if not silent:
        print("Procesando velas...", end=" ", flush=True)

    for candle in clock:
        last_candle = candle
        signal = strategy._tick(candle, wallet)

        if not signal.is_actionable:
            continue

        order_side = signal.to_order_side()

        risk_reason = risk.check(order_side, signal.price, wallet, candle)
        if risk_reason:
            n_ignorados += 1
            ign_motivos[risk_reason] = ign_motivos.get(risk_reason, 0) + 1
            wallet.update(TradeRecord(
                ts=candle.ts, side=order_side.value, price=signal.price,
                ignored=True, ignore_reason=risk_reason,
            ))
            continue

        order = ob.execute_with_guards(
            order_side, signal.price, wallet, candle_ts=candle.ts
        )

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

        if wallet.get_trade_log():
            lt = wallet.get_trade_log()[-1]
            lt["score_bot"] = signal.score if signal.side == SignalSide.BUY  else 0.0
            lt["score_top"] = signal.score if signal.side == SignalSide.SELL else 0.0

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(wallet, candle.ts, candle.close))

    if not silent:
        print(f"OK  ({clock.total_candles:,} velas)")

    strategy.on_stop(wallet)

    if last_candle is None:
        return {}

    precio_final = last_candle.close
    port_final   = wallet.portfolio_value(precio_final)
    pnl_pct      = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100

    first_c  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_i = first_c[0].close if first_c else precio_final
    bh_pnl   = (precio_final / precio_i - 1) * 100

    all_r = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl   = min(c.low  for c in all_r) if all_r else 0
    ath   = max(c.high for c in all_r) if all_r else 0

    summary = {
        "estrategia"              : strategy.name,
        "fecha_inicio"            : CL.FECHA_INICIO,
        "fecha_fin"               : CL.FECHA_FIN,
        "saldo_inicial_usdt"      : CL.SALDO_USDT_INICIAL,
        "usdt_balance_final"      : round(wallet.get_usdt_balance(), 8),
        "btc_balance_final"       : round(wallet.get_btc_balance(), 10),
        "btc_acumulado_total"     : round(wallet.get_btc_acumulado(), 10),
        "btc_en_posiciones_final" : round(wallet.btc_en_posiciones(), 10),
        "precio_promedio_final"   : round(wallet.precio_promedio_posiciones(), 8),
        "portfolio_value_final"   : round(port_final, 4),
        "pnl_pct"                 : round(pnl_pct, 4),
        "buy_hold_pnl_pct"        : round(bh_pnl, 4),
        "alpha_vs_bh"             : round(pnl_pct - bh_pnl, 4),
        "precio_min_comprado"     : round(precio_min_comprado, 4) if n_compras else None,
        "precio_max_vendido"      : round(precio_max_vendido,  4) if n_ventas  else None,
        "atl_final"               : round(atl, 4),
        "ath_proyectado_final"    : round(ath, 4),
        "total_trades_ejecutados" : n_compras + n_ventas,
        "total_compras"           : n_compras,
        "total_ventas"            : n_ventas,
        "total_ignorados"         : n_ignorados,
        "ordenes_canceladas"      : 0,
        "ignorados_por_motivo"    : ign_motivos,
        "positions_count_final"   : wallet.positions_count,
        "usdt_reserva_aplicada"   : 0.0,
        "umbral_filtro"           : ctx_min_pts,
        "parametros"              : strategy.describe(),
    }

    wallet.flush(summary)

    if not silent:
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
        print(f"  Ignorados        : {n_ignorados:,}")
        print(f"  Posiciones abier.: {wallet.positions_count}")
        print(f"  Tiempo total     : {elapsed:.1f}s")
        print(sep)
        print(f"\n✓ Resultado guardado en: {json_path}")

    return summary


# ════════════════════════════════════════════════════════════════════
# GRID SEARCH — varía indicadores Y umbrales
# ════════════════════════════════════════════════════════════════════

def run_grid_search(top_n: int = 20) -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║      GRID SEARCH — DUAL REVERSAL                        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Período: {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print("─" * 60)

    # ── Espacio de búsqueda ───────────────────────────────────────
    # Indicadores de señal
    grid_rsi_period     = [24, 14, 20]
    grid_ma_short       = [30, 20, 25]
    grid_ma_long        = [70, 50, 60]
    grid_window         = [68, 24, 32]
    grid_last_n         = [12, 5, 7]

    # Contexto (umbrales RSI y MA20 simétricos)
    grid_ctx_rsi_buy    = [35.0, 40.0, 30.0]
    # ctx_rsi_sell = 100 - ctx_rsi_buy para mantener simetría exacta
    grid_ctx_ma20_dist  = [1.5, 2.0, 3.0]   # |distancia| — positivo para SELL, negativo para BUY
    grid_ctx_min_pts    = [5, 6, 4]

    # Disparador
    grid_trig_rsi_slope = [0.02, 0.03, 0.05]
    grid_trig_wick      = [0.25, 0.28, 0.32]
    grid_trig_min_pts   = [4, 5]

    total = (len(grid_rsi_period) * len(grid_ma_short) * len(grid_ma_long) *
             len(grid_window) * len(grid_last_n) *
             len(grid_ctx_rsi_buy) * len(grid_ctx_ma20_dist) * len(grid_ctx_min_pts) *
             len(grid_trig_rsi_slope) * len(grid_trig_wick) * len(grid_trig_min_pts))

    print(f"  Espacio total: {total:,} combinaciones")
    print()
    print("  Parámetros explorados:")
    print(f"    rsi_period    : {grid_rsi_period}")
    print(f"    ma_short      : {grid_ma_short}")
    print(f"    ma_long       : {grid_ma_long}")
    print(f"    window        : {grid_window}")
    print(f"    last_n        : {grid_last_n}")
    print(f"    ctx_rsi_buy   : {grid_ctx_rsi_buy}  (sell = 100 - buy)")
    print(f"    ctx_ma20 dist : {grid_ctx_ma20_dist}  (buy negativo, sell positivo)")
    print(f"    ctx_min_pts   : {grid_ctx_min_pts}")
    print(f"    trig_rsi_slope: {grid_trig_rsi_slope}")
    print(f"    trig_wick     : {grid_trig_wick}")
    print(f"    trig_min_pts  : {grid_trig_min_pts}")
    print()

    # Opción de muestreo aleatorio si el espacio es muy grande
    import random
    MAX_COMBOS = 500
    if total > MAX_COMBOS:
        print(f"  Espacio demasiado grande — muestreando {MAX_COMBOS} combinaciones aleatorias")
        print()
        random.seed(42)
        combos = []
        for _ in range(MAX_COMBOS):
            combos.append({
                "rsi_period":     random.choice(grid_rsi_period),
                "ma_short":       random.choice(grid_ma_short),
                "ma_long":        random.choice(grid_ma_long),
                "window":         random.choice(grid_window),
                "last_n":         random.choice(grid_last_n),
                "ctx_rsi_buy":    random.choice(grid_ctx_rsi_buy),
                "ctx_ma20_dist":  random.choice(grid_ctx_ma20_dist),
                "ctx_min_pts":    random.choice(grid_ctx_min_pts),
                "trig_rsi_slope": random.choice(grid_trig_rsi_slope),
                "trig_wick":      random.choice(grid_trig_wick),
                "trig_min_pts":   random.choice(grid_trig_min_pts),
            })
    else:
        combos = [
            {"rsi_period": rp, "ma_short": ms, "ma_long": ml,
             "window": w, "last_n": ln,
             "ctx_rsi_buy": crb, "ctx_ma20_dist": cmd, "ctx_min_pts": cmp,
             "trig_rsi_slope": trs, "trig_wick": tw, "trig_min_pts": tmp}
            for rp in grid_rsi_period
            for ms in grid_ma_short
            for ml in grid_ma_long
            for w  in grid_window
            for ln in grid_last_n
            for crb in grid_ctx_rsi_buy
            for cmd in grid_ctx_ma20_dist
            for cmp in grid_ctx_min_pts
            for trs in grid_trig_rsi_slope
            for tw  in grid_trig_wick
            for tmp in grid_trig_min_pts
        ]

    resultados = []
    t0_grid = time.time()

    for i, p in enumerate(combos, 1):
        # Filtro de consistencia: MA_SHORT < MA_LONG y LAST_N < WINDOW
        if p["ma_short"] >= p["ma_long"]:
            continue
        if p["last_n"] >= p["window"]:
            continue

        res = run_backtest(
            rsi_period     = p["rsi_period"],
            ma_short       = p["ma_short"],
            ma_long        = p["ma_long"],
            window         = p["window"],
            last_n         = p["last_n"],
            ctx_rsi_buy    = p["ctx_rsi_buy"],
            ctx_rsi_sell   = 100.0 - p["ctx_rsi_buy"],  # simetría exacta
            ctx_ma20_buy   = -p["ctx_ma20_dist"],
            ctx_ma20_sell  =  p["ctx_ma20_dist"],
            ctx_min_pts    = p["ctx_min_pts"],
            trig_rsi_slope = p["trig_rsi_slope"],
            trig_wick      = p["trig_wick"],
            trig_min_pts   = p["trig_min_pts"],
            results_json   = "/tmp/dr_grid_tmp.json",
            silent         = True,
        )

        if res:
            resultados.append({
                **p,
                "pnl":     res["pnl_pct"],
                "alpha":   res["alpha_vs_bh"],
                "bh":      res["buy_hold_pnl_pct"],
                "compras": res["total_compras"],
                "ventas":  res["total_ventas"],
                "pos_fin": res["positions_count_final"],
            })

        if i % 50 == 0:
            elapsed = time.time() - t0_grid
            eta = elapsed / i * (len(combos) - i)
            print(f"  {i}/{len(combos)}  ({elapsed:.0f}s transcurridos  ETA ≈{eta:.0f}s)",
                  flush=True)

    resultados.sort(key=lambda r: r["alpha"], reverse=True)

    # ── Mostrar resultados ────────────────────────────────────────
    elapsed_total = time.time() - t0_grid
    print(f"\n  Grid completo en {elapsed_total:.0f}s — {len(resultados)} combinaciones válidas")
    print()

    hdr = (f"  {'RSI':>4} {'MAs':>7} {'W':>3} {'N':>2} "
           f"{'rsi_b':>6} {'ma20':>5} {'ctx':>4} "
           f"{'rsl':>5} {'wk':>5} {'trg':>4} "
           f"{'PnL%':>8} {'Alpha':>8} {'B':>5} {'V':>5}")
    print(hdr)
    print(f"  {'─'*len(hdr.strip())}")

    for r in resultados[:top_n]:
        mas = f"{r['ma_short']}/{r['ma_long']}"
        print(
            f"  {r['rsi_period']:>4} {mas:>7} {r['window']:>3} {r['last_n']:>2} "
            f"  {r['ctx_rsi_buy']:>5.0f} {r['ctx_ma20_dist']:>5.1f} {r['ctx_min_pts']:>4} "
            f"  {r['trig_rsi_slope']:>5.2f} {r['trig_wick']:>5.2f} {r['trig_min_pts']:>4} "
            f"  {r['pnl']:>+8.2f}% {r['alpha']:>+8.2f}%"
            f"  {r['compras']:>5} {r['ventas']:>5}"
        )

    if resultados:
        best = resultados[0]
        print(f"\n  ─── Mejor combinación (alpha={best['alpha']:+.2f}%) ───")
        print(f"    Indicadores  : RSI({best['rsi_period']})  "
              f"MA{best['ma_short']}/{best['ma_long']}  "
              f"W={best['window']}  N={best['last_n']}")
        print(f"    Contexto     : RSI_buy<{best['ctx_rsi_buy']:.0f}  "
              f"MA20_dist={best['ctx_ma20_dist']:.1f}%  "
              f"min={best['ctx_min_pts']}pts")
        print(f"    Disparador   : rsi_slope>{best['trig_rsi_slope']:.2f}  "
              f"wick>{best['trig_wick']:.2f}  "
              f"min={best['trig_min_pts']}pts")
        print(f"    PnL={best['pnl']:+.2f}%  Alpha={best['alpha']:+.2f}%  "
              f"B={best['compras']}  V={best['ventas']}")

        # Guardar ranking completo en JSON
        import json
        ranking_path = "grid_results_dual_reversal.json"
        with open(ranking_path, "w") as f:
            json.dump(resultados, f, indent=2)
        print(f"\n  Ranking completo guardado en: {ranking_path}")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest Dual Reversal — 2 capas calibradas BTC/USDT"
    )
    parser.add_argument("--grid",  action="store_true",
                        help="Grid search completo de parámetros")
    parser.add_argument("--top",   type=int, default=20,
                        help="Cuántas combinaciones mostrar (default: 20)")
    args = parser.parse_args()

    if args.grid:
        run_grid_search(top_n=args.top)
    else:
        run_backtest()
