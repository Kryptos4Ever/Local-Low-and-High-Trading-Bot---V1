"""
backtest_zigzag_reversal.py — Runner de la Estrategia ZigZag Reversal
══════════════════════════════════════════════════════════════════════
Backtest de ZigZagReversalStrategy: reversión a la media calibrada
empíricamente sobre los turning points del ZigZag 5%.

Uso:
    python backtest_zigzag_reversal.py           # run con parámetros default
    python backtest_zigzag_reversal.py --grid    # grid search de parámetros

Configuración del entorno:
    config_local.py  →  rutas, fechas, capital, comisión, MAX_POSICIONES
"""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL

from actors.price_feed              import SQLiteFeed
from actors.wallet                  import JSONWallet, MemoryWallet, TradeRecord
from actors.order_book              import SimulatedOrderBook, OrderSide
from actors.clock                   import LocalClock
from risk.risk_manager              import RiskManager, RiskConfig
from state.state_manager            import MemoryStateManager, Checkpoint
from strategies.zigzag_reversal     import ZigZagReversalStrategy
from strategies.base_strategy       import SignalSide
from support.logger                 import get_logger

log = get_logger("backtest_zigzag_reversal")


# ════════════════════════════════════════════════════════════════════
# PARÁMETROS DE LA ESTRATEGIA
# ════════════════════════════════════════════════════════════════════

# Indicadores
RSI_PERIOD      = 14
MA_SHORT        = 20
MA_LONG         = 50
WINDOW          = 24     # velas de historia para las features del segmento
LAST_N          = 5      # velas para slopes y ratios finales

# Umbrales BUY (calibrados sobre p25/p50 de BOTTOMs)
RSI_BUY_STRONG  = 30.0   # p35 de BOTTOMs → 3 pts
RSI_BUY_WEAK    = 40.0   # p75 de BOTTOMs → 2 pts
MA20_BUY_STRONG = -3.5   # p25 de BOTTOMs → 2 pts
MA20_BUY_WEAK   = -1.0   # p75 de BOTTOMs → 1 pt

# Umbrales SELL (calibrados sobre p25/p50 de TOPs)
RSI_SELL_STRONG = 65.0   # p50 de TOPs    → 3 pts
RSI_SELL_WEAK   = 55.0   # p25 de TOPs    → 2 pts
MA20_SELL_STRONG= 2.5    # p75 de TOPs    → 2 pts
MA20_SELL_WEAK  = 1.0    # p25 de TOPs    → 1 pt

# Puntaje mínimo (rango razonable: 4-7)
MIN_PUNTOS_BUY  = 5
MIN_PUNTOS_SELL = 5

# Gestión de riesgo
STOP_LOSS_PCT   = 0.04   # 4% — p25 del move_pct de los turning points es 5.4%


# ════════════════════════════════════════════════════════════════════
# RUNNER
# ════════════════════════════════════════════════════════════════════

def run_backtest(
    min_puntos_buy:  int   = MIN_PUNTOS_BUY,
    min_puntos_sell: int   = MIN_PUNTOS_SELL,
    stop_loss_pct:   float = STOP_LOSS_PCT,
    rsi_buy_strong:  float = RSI_BUY_STRONG,
    rsi_sell_strong: float = RSI_SELL_STRONG,
    results_json:    str   = None,
    silent:          bool  = False,
) -> dict:
    t_start   = time.time()
    json_path = results_json or CL.RESULTS_JSON

    if not silent:
        print("╔══════════════════════════════════════════════════════════╗")
        print("║    BACKTEST ZIGZAG REVERSAL — Calibrado BTC/USDT        ║")
        print("╚══════════════════════════════════════════════════════════╝")
        print(f"  Rango         : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
        print(f"  Capital       : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
        print(f"  Max posiciones: {CL.MAX_POSICIONES}")
        print(f"  Comisión      : {CL.COMMISSION_PCT}%")
        print(f"  Puntos BUY/SELL: {min_puntos_buy}/{min_puntos_sell}")
        print(f"  RSI BUY/SELL  : <{rsi_buy_strong}/<{rsi_sell_strong}")
        print(f"  Stop-loss     : {stop_loss_pct*100:.1f}%")
        print(f"  Output JSON   : {json_path}")
        print("─" * 60)

    # ── 1. Actores ─────────────────────────────────────────────────
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

    strategy = ZigZagReversalStrategy(
        rsi_period       = RSI_PERIOD,
        ma_short         = MA_SHORT,
        ma_long          = MA_LONG,
        window           = WINDOW,
        last_n           = LAST_N,
        rsi_buy_strong   = rsi_buy_strong,
        rsi_buy_weak     = RSI_BUY_WEAK,
        ma20_buy_strong  = MA20_BUY_STRONG,
        ma20_buy_weak    = MA20_BUY_WEAK,
        rsi_sell_strong  = rsi_sell_strong,
        rsi_sell_weak    = RSI_SELL_WEAK,
        ma20_sell_strong = MA20_SELL_STRONG,
        ma20_sell_weak   = MA20_SELL_WEAK,
        min_puntos_buy   = min_puntos_buy,
        min_puntos_sell  = min_puntos_sell,
        stop_loss_pct    = stop_loss_pct,
    )

    # ── 2. on_start ────────────────────────────────────────────────
    strategy.on_start(wallet)

    clock = LocalClock(
        feed   = feed,
        start  = CL.FECHA_INICIO,
        end    = CL.FECHA_FIN,
        symbol = CL.SYMBOL,
    )

    # ── 3. Contadores ──────────────────────────────────────────────
    n_compras = n_ventas = n_ignorados = n_stoploss = 0
    ign_motivos: dict[str, int] = {}
    precio_min_comprado = float("inf")
    precio_max_vendido  = float("-inf")
    last_candle = None

    # ── 4. Loop principal ──────────────────────────────────────────
    if not silent:
        print("Procesando velas...", end=" ", flush=True)

    for candle in clock:
        last_candle = candle
        signal      = strategy._tick(candle, wallet)

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
                if signal.score == 0.0:
                    n_stoploss += 1
        else:
            n_ignorados += 1
            motivo = order.reject_reason or "desconocido"
            ign_motivos[motivo] = ign_motivos.get(motivo, 0) + 1

        if wallet.get_trade_log():
            last_trade = wallet.get_trade_log()[-1]
            last_trade["score_bot"] = signal.score if signal.side == SignalSide.BUY  else 0.0
            last_trade["score_top"] = signal.score if signal.side == SignalSide.SELL else 0.0

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(
            wallet, candle.ts, candle.close,
            metadata={"estrategia": strategy.name},
        ))

    if not silent:
        print(f"OK  ({clock.total_candles:,} velas)")

    # ── 5. Métricas finales ────────────────────────────────────────
    strategy.on_stop(wallet)

    if last_candle is None:
        return {}

    precio_final = last_candle.close
    port_final   = wallet.portfolio_value(precio_final)
    pnl_pct      = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100

    first_c      = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_i     = first_c[0].close if first_c else precio_final
    bh_pnl       = (precio_final / precio_i - 1) * 100

    all_range = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl = min(c.low  for c in all_range) if all_range else 0
    ath = max(c.high for c in all_range) if all_range else 0

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
        "total_stoploss"          : n_stoploss,
        "total_ignorados"         : n_ignorados,
        "ordenes_canceladas"      : 0,
        "ignorados_por_motivo"    : ign_motivos,
        "positions_count_final"   : wallet.positions_count,
        "usdt_reserva_aplicada"   : 0.0,
        "umbral_filtro"           : min_puntos_buy,
        "parametros"              : strategy.describe(),
    }

    wallet.flush(summary)

    elapsed = time.time() - t_start
    sign    = "+" if pnl_pct >= 0 else ""
    if not silent:
        sep = "═" * 60
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
        print(f"  Ventas           : {n_ventas:,}  (stop-loss: {n_stoploss})")
        print(f"  Ignorados        : {n_ignorados:,}")
        print(f"  Posiciones abier.: {wallet.positions_count}")
        print(f"  Tiempo total     : {elapsed:.1f}s")
        print(sep)
        print(f"\n✓ Resultado guardado en: {json_path}")

    return summary


# ════════════════════════════════════════════════════════════════════
# GRID SEARCH
# ════════════════════════════════════════════════════════════════════

def run_grid_search() -> None:
    print("╔══════════════════════════════════════════════════════════╗")
    print("║       GRID SEARCH — ZIGZAG REVERSAL                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Período: {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print("─" * 60)

    grid = {
        "min_puntos_buy":  [4, 5, 6],
        "min_puntos_sell": [4, 5, 6],
        "stop_loss_pct":   [0.0, 0.03, 0.04, 0.05],
        "rsi_buy_strong":  [28.0, 30.0, 33.0],
        "rsi_sell_strong": [62.0, 65.0, 68.0],
    }

    combos = list(itertools.product(*grid.values()))
    keys   = list(grid.keys())
    print(f"  {len(combos)} combinaciones a testear\n")

    resultados = []
    for i, combo in enumerate(combos, 1):
        params = dict(zip(keys, combo))
        res = run_backtest(
            **params,
            results_json = "/tmp/zz_grid_tmp.json",
            silent       = True,
        )
        if res:
            resultados.append({**params,
                                "pnl":     res["pnl_pct"],
                                "alpha":   res["alpha_vs_bh"],
                                "bh":      res["buy_hold_pnl_pct"],
                                "compras": res["total_compras"],
                                "ventas":  res["total_ventas"],
                                "stops":   res["total_stoploss"]})
        if i % 20 == 0:
            print(f"  Progreso: {i}/{len(combos)}", flush=True)

    resultados.sort(key=lambda r: r["alpha"], reverse=True)

    hdr = f"  {'pBUY':>5}  {'pSEL':>5}  {'SL%':>5}  {'RSI_B':>6}  {'RSI_S':>6}  {'PnL%':>8}  {'Alpha':>8}  {'B':>5}  {'V':>5}  {'SL':>4}"
    print(f"\n{'─'*len(hdr)}")
    print(hdr)
    print(f"{'─'*len(hdr)}")
    for r in resultados[:20]:
        sl_s = f"{r['stop_loss_pct']*100:.0f}%" if r['stop_loss_pct'] > 0 else "  —  "
        print(f"  {r['min_puntos_buy']:>5}  {r['min_puntos_sell']:>5}  {sl_s:>5}  "
              f"{r['rsi_buy_strong']:>6.0f}  {r['rsi_sell_strong']:>6.0f}  "
              f"{r['pnl']:>+8.2f}%  {r['alpha']:>+8.2f}%  "
              f"{r['compras']:>5}  {r['ventas']:>5}  {r['stops']:>4}")
    print(f"{'─'*len(hdr)}")
    if resultados:
        best = resultados[0]
        print(f"\nMejor combinación (por alpha):")
        for k, v in best.items():
            if k not in ('pnl', 'alpha', 'bh', 'compras', 'ventas', 'stops'):
                print(f"  {k} = {v}")
        print(f"  → PnL={best['pnl']:+.2f}%  Alpha={best['alpha']:+.2f}%")


# ════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Backtest ZigZag Reversal — Calibrado sobre turning points"
    )
    parser.add_argument("--grid", action="store_true",
                        help="Ejecutar grid search de parámetros")
    args = parser.parse_args()

    if args.grid:
        run_grid_search()
    else:
        run_backtest()
