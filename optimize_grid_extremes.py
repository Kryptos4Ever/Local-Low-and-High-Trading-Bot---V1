"""
optimize_grid_extremes.py — Optimizador de GridExtremesStrategy
════════════════════════════════════════════════════════════════
Barre todas las combinaciones posibles de parámetros y guarda las
top-15 configuraciones en un JSON compatible con el resto del sistema.

Uso:
    python optimize_grid_extremes.py

Configuración:
    Editar los grids en la sección ESPACIOS DE BÚSQUEDA más abajo.
    Los demás parámetros (capital, fechas, comisión) se leen de config_local.py.

Salida:
    optimize_grid_best.json   — top-15 configuraciones ordenadas por score
"""

from __future__ import annotations

import json
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent))


# ══════════════════════════════════════════════════════════════════════════════
# ESPACIOS DE BÚSQUEDA — editar para ajustar rangos
# ══════════════════════════════════════════════════════════════════════════════

GRID = {
    "max_posiciones": [2, 4, 5, 6, 7, 8, 9, 10],
    "ventana":        [4, 6, 8, 10],
    "drop_pct_buy":   [4.0, 5.0, 6.0, 8.0, 10.0],
    "rise_pct_sell":  [4.0, 5.0, 6.0, 8.0, 10.0],
    "retroactive":    [False, True],
}

TOP_N = 15   # cuántos resultados guardar en el JSON


# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST
# ══════════════════════════════════════════════════════════════════════════════

def run_backtest(
    max_pos: int,
    ventana: int,
    drop:    float,
    rise:    float,
    retro:   bool,
) -> Dict:
    import config_local as CL
    from actors.price_feed        import SQLiteFeed
    from actors.wallet            import MemoryWallet, TradeRecord
    from actors.order_book        import SimulatedOrderBook, OrderSide
    from actors.clock             import LocalClock
    from risk.risk_manager        import RiskManager, RiskConfig
    from strategies.grid_extremes import GridExtremesStrategy

    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    wallet = MemoryWallet(usdt_inicial=CL.SALDO_USDT_INICIAL,
                          max_posiciones=max_pos)
    ob     = SimulatedOrderBook(commission_pct=CL.COMMISSION_PCT,
                                max_posiciones=max_pos)
    risk   = RiskManager(config=RiskConfig.permissive(),
                         usdt_inicial=CL.SALDO_USDT_INICIAL)
    clock  = LocalClock(feed=feed, start=CL.FECHA_INICIO,
                        end=CL.FECHA_FIN, symbol=CL.SYMBOL)
    strat  = GridExtremesStrategy(ventana=ventana, drop_pct_buy=drop,
                                   rise_pct_sell=rise, retroactive=retro)
    # Sobreescribir _max_levels para que la estrategia use el max_pos del grid
    strat._max_levels = max_pos
    strat.on_start(wallet)

    n_buy = n_sell = wins = 0
    buy_prices: list = []
    last_candle = None

    for candle in clock:
        last_candle = candle
        signal = strat._tick(candle, wallet)
        if not signal.is_actionable:
            continue
        side = signal.to_order_side()
        rr   = risk.check(side, signal.price, wallet, candle)
        if rr:
            wallet.update(TradeRecord(ts=candle.ts, side=side.value,
                                      price=signal.price, ignored=True,
                                      ignore_reason=rr))
            continue
        order = ob.execute_with_guards(side, signal.price, wallet,
                                       candle_ts=candle.ts)
        if order.is_filled:
            if side == OrderSide.BUY:
                n_buy += 1
                buy_prices.append(signal.price)
            else:
                n_sell += 1
                if buy_prices:
                    wins += 1 if signal.price > sum(buy_prices)/len(buy_prices) else 0
                    buy_prices.clear()
        risk.update_peak(wallet.portfolio_value(candle.close))

    if last_candle is None:
        return {}

    first = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    p_ini = first[0].close if first else last_candle.close
    port  = wallet.portfolio_value(last_candle.close)
    pnl   = (port / CL.SALDO_USDT_INICIAL - 1) * 100
    bh    = (last_candle.close / p_ini - 1) * 100

    return {
        "max_posiciones": max_pos,
        "ventana":        ventana,
        "drop_pct_buy":   drop,
        "rise_pct_sell":  rise,
        "retroactive":    retro,
        "pnl_pct":        round(pnl, 2),
        "bh_pnl":         round(bh, 2),
        "alpha_vs_bh":    round(pnl - bh, 2),
        "win_rate":       round(wins / max(n_sell, 1), 4),
        "n_compras":      n_buy,
        "n_ventas":       n_sell,
        "score":          round(pnl * 0.45 + (pnl - bh) * 0.35
                                + wins / max(n_sell, 1) * 100 * 0.20, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    import config_local as CL

    combos = list(product(
        GRID["max_posiciones"],
        GRID["ventana"],
        GRID["drop_pct_buy"],
        GRID["rise_pct_sell"],
        GRID["retroactive"],
    ))
    total = len(combos)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║      OPTIMIZADOR — GridExtremesStrategy BTC/USDT        ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Rango     : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital   : ${CL.SALDO_USDT_INICIAL:,.2f} USDT  |  "
          f"Comisión: {CL.COMMISSION_PCT}%")
    print(f"  Total combinaciones: {total}")
    print("─" * 60)

    results: List[Dict] = []
    best_score = -9999.0
    t0 = time.time()

    for i, (mp, v, d, r, retro) in enumerate(combos, 1):
        res = run_backtest(mp, v, d, r, retro)
        if not res:
            continue

        results.append(res)
        results.sort(key=lambda x: x["score"], reverse=True)
        results = results[:TOP_N]

        is_best = res["score"] > best_score
        if is_best:
            best_score = res["score"]

        elapsed = time.time() - t0
        eta     = elapsed / i * (total - i)
        marker  = "★" if is_best else " "
        print(
            f"  [{i:5d}/{total}] {marker} "
            f"mp={mp}  v={v:2d}  drop={d:5.1f}%  rise={r:5.1f}%  "
            f"retro={str(retro):<5}  →  "
            f"PnL={res['pnl_pct']:+7.2f}%  "
            f"alpha={res['alpha_vs_bh']:+7.2f}%  "
            f"wr={res['win_rate']:.1%}  "
            f"score={res['score']:7.2f}  "
            f"ETA:{eta:5.0f}s"
        )

    # Guardar resultado
    out = Path("optimize_grid_best.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Resumen final
    elapsed = time.time() - t0
    sep = "═" * 66
    print(f"\n{sep}")
    print(f"  TOP-{TOP_N} CONFIGURACIONES")
    print(sep)
    print(f"  {'#':>3}  {'mp':>3}  {'v':>3}  {'drop%':>6}  {'rise%':>6}  "
          f"{'retro':<5}  {'PnL%':>7}  {'alpha':>7}  {'wr':>6}  {'score':>7}")
    print("  " + "─" * 62)
    for i, r in enumerate(results, 1):
        print(
            f"  {i:3d}  "
            f"{r['max_posiciones']:>3}  "
            f"{r['ventana']:>3}  "
            f"{r['drop_pct_buy']:>6.1f}  "
            f"{r['rise_pct_sell']:>6.1f}  "
            f"{str(r['retroactive']):<5}  "
            f"{r['pnl_pct']:>+7.2f}  "
            f"{r['alpha_vs_bh']:>+7.2f}  "
            f"{r['win_rate']:>6.1%}  "
            f"{r['score']:>7.2f}"
        )
    print(sep)
    if results:
        best = results[0]
        print(f"\n  GANADOR → copiar en backtest_grid_extremes.py y config_local.py:")
        print(f"    MAX_POSICIONES = {best['max_posiciones']}")
        print(f"    VENTANA        = {best['ventana']}")
        print(f"    DROP_PCT_BUY   = {best['drop_pct_buy']}")
        print(f"    RISE_PCT_SELL  = {best['rise_pct_sell']}")
        print(f"    RETROACTIVE    = {best['retroactive']}")
    print(f"\n  {total} combinaciones en {elapsed:.1f}s  "
          f"({elapsed/total:.2f}s por backtest)")
    print(f"  Guardado en: {out}\n")


if __name__ == "__main__":
    main()