"""
backtest_compuesto.py — Runner de la Señal Compuesta
══════════════════════════════════════════════════════
Ensambla todos los actores y corre CompuestoStrategy
(DNA + Lyapunov + PE + Delta → Score 0-100).

Resultado: JSON compatible con Graficador_v2.py

Uso:
    python backtest_compuesto.py          # calcula todo desde cero
    python backtest_compuesto.py --fast   # usa cache .cache_compuesto/
    python backtest_compuesto.py --nocache # borra cache y recalcula

Configuración:
    · config_local.py        → rutas, fechas, capital
    · mode_config.py         → modos (todos en False para backtest local)
    · strategies/compuesto.py → parámetros de la estrategia (inline abajo)
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import config_local as CL
import mode_config  as MC

from actors.price_feed    import SQLiteFeed
from actors.wallet        import JSONWallet
from actors.order_book    import SimulatedOrderBook, OrderSide
from actors.clock         import LocalClock
from actors.wallet        import TradeRecord
from risk.risk_manager    import build_risk_manager
from state.state_manager  import MemoryStateManager, Checkpoint
from strategies.compuesto import CompuestoStrategy
from strategies.base_strategy import SignalSide
from support.logger       import get_logger

log = get_logger("backtest_compuesto")


# ══════════════════════════════════════════════════════════════════════════════
# PARÁMETROS DE LA ESTRATEGIA
# (viven aquí — cambiables sin tocar la clase)
# ══════════════════════════════════════════════════════════════════════════════
THR_BOT        = 75.0
THR_TOP        = 75.0
COOLDOWN       = 16
SUAVIZADO      = 6
VENTANA_SCORE  = 500
CACHE_DIR      = ".cache_compuesto"


# ══════════════════════════════════════════════════════════════════════════════
# RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def main(use_cache: bool = True, clear_cache: bool = False) -> None:
    t_start = time.time()

    # Manejar cache
    if clear_cache and Path(CACHE_DIR).exists():
        shutil.rmtree(CACHE_DIR)
        print(f"✓ Cache borrado: {CACHE_DIR}")

    print("╔══════════════════════════════════════════════════════════╗")
    print("║    BACKTEST COMPUESTO — DNA + Lyapunov + PE + Delta     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Rango         : {CL.FECHA_INICIO} → {CL.FECHA_FIN}")
    print(f"  Capital       : ${CL.SALDO_USDT_INICIAL:,.2f} USDT")
    print(f"  Max posiciones: {CL.MAX_POSICIONES}")
    print(f"  Comisión      : {CL.COMMISSION_PCT}%")
    print(f"  Umbral bot/top: {THR_BOT} / {THR_TOP}")
    print(f"  Cooldown      : {COOLDOWN}h  Suavizado: {SUAVIZADO}")
    print(f"  Ventana score : {VENTANA_SCORE}")
    print(f"  Cache         : {'--fast (reutiliza)' if use_cache else 'recalcula todo'}")
    print(f"  Output JSON   : {CL.RESULTS_JSON}")
    print("─" * 60)

    # ── 1. Construir actores ───────────────────────────────────────────────────
    feed   = SQLiteFeed(db_path=CL.DB_PATH, table=CL.DB_TABLE)
    wallet = JSONWallet(
        usdt_inicial   = CL.SALDO_USDT_INICIAL,
        max_posiciones = CL.MAX_POSICIONES,
        json_path      = CL.RESULTS_JSON,
    )
    ob     = SimulatedOrderBook(
        commission_pct = CL.COMMISSION_PCT,
        max_posiciones = CL.MAX_POSICIONES,
    )
    risk   = build_risk_manager(usdt_inicial=CL.SALDO_USDT_INICIAL)
    state  = MemoryStateManager()

    strategy = CompuestoStrategy(
        thr_bot        = THR_BOT,
        thr_top        = THR_TOP,
        cooldown       = COOLDOWN,
        suavizado      = SUAVIZADO,
        ventana_score  = VENTANA_SCORE,
        cache_dir      = CACHE_DIR,
        force_recompute= not use_cache,
    )

    # ── 2. on_start: carga/calcula scores ─────────────────────────────────────
    # Pasar el feed completo (dataset completo, no solo el rango de backtest)
    # para que el score tenga warm-up suficiente
    print("Inicializando estrategia (puede tardar en primer run)...")
    strategy.on_start(
        wallet = wallet,
        feed   = feed,
        start  = "2017-01-01",   # dataset completo para warm-up
        end    = "2030-01-01",
        symbol = CL.SYMBOL,
    )

    # Guardar timestamps en cache para futuros runs
    import numpy as np
    all_candles = feed.get_candles("2017-01-01", "2030-01-01", CL.SYMBOL)
    ts_arr = np.array([c.ts for c in all_candles], dtype=np.int64)
    np.save(Path(CACHE_DIR) / "timestamps.npy", ts_arr)

    # ── 3. Clock solo para el rango de backtest ────────────────────────────────
    clock = LocalClock(feed, start=CL.FECHA_INICIO, end=CL.FECHA_FIN,
                       symbol=CL.SYMBOL)

    # ── 4. Contadores ─────────────────────────────────────────────────────────
    n_compras   = 0
    n_ventas    = 0
    n_ignorados = 0
    ign_motivos: dict[str, int] = {}
    precio_min_comprado = float("inf")
    precio_max_vendido  = float("-inf")
    last_candle = None

    # ── 5. Loop principal ─────────────────────────────────────────────────────
    print("Procesando velas...", end=" ", flush=True)

    for candle in clock:
        last_candle = candle
        signal = strategy._tick(candle, wallet)

        if not signal.is_actionable:
            continue

        order_side = signal.to_order_side()

        # Risk check
        risk_reason = risk.check(order_side, signal.price, wallet, candle)
        if risk_reason:
            n_ignorados += 1
            ign_motivos[risk_reason] = ign_motivos.get(risk_reason, 0) + 1
            wallet.update(TradeRecord(
                ts=candle.ts, side=order_side.value, price=signal.price,
                ignored=True, ignore_reason=risk_reason,
            ))
            continue

        order = ob.execute_with_guards(order_side, signal.price, wallet, candle_ts=candle.ts)

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

        # Enriquecer trade con scores para el JSON / Graficador
        if wallet.get_trade_log():
            last_trade = wallet.get_trade_log()[-1]
            last_trade["score_bot"] = signal.score if signal.side == SignalSide.BUY  else 0.0
            last_trade["score_top"] = signal.score if signal.side == SignalSide.SELL else 0.0

        risk.update_peak(wallet.portfolio_value(candle.close))
        state.save(Checkpoint.from_wallet(
            wallet, candle.ts, candle.close,
            metadata={"estrategia": strategy.name},
        ))

    print(f"OK  ({clock.total_candles:,} velas)")

    # ── 6. Métricas finales ───────────────────────────────────────────────────
    strategy.on_stop(wallet)

    if last_candle is None:
        print("✗ No se encontraron velas en el rango indicado.")
        return

    precio_final = last_candle.close
    port_final   = wallet.portfolio_value(precio_final)
    pnl_pct      = (port_final / CL.SALDO_USDT_INICIAL - 1) * 100

    first_c  = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_INICIO)
    precio_i = first_c[0].close if first_c else precio_final
    bh_pnl   = (precio_final / precio_i - 1) * 100

    all_range   = feed.get_candles(CL.FECHA_INICIO, CL.FECHA_FIN)
    atl = min(c.low  for c in all_range) if all_range else 0
    ath = max(c.high for c in all_range) if all_range else 0

    summary = {
        "estrategia":               strategy.name,
        "fecha_inicio":             CL.FECHA_INICIO,
        "fecha_fin":                CL.FECHA_FIN,
        "saldo_inicial_usdt":       CL.SALDO_USDT_INICIAL,
        "usdt_balance_final":       round(wallet.get_usdt_balance(), 8),
        "btc_balance_final":        round(wallet.get_btc_balance(), 10),
        "btc_acumulado_total":      round(wallet.get_btc_balance(), 10),
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
        "umbral_filtro":            THR_BOT,
        "parametros":               strategy.describe(),
    }

    wallet.flush(summary)

    # ── 7. Resumen en consola ─────────────────────────────────────────────────
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
    print(f"  Tiempo total     : {elapsed:.1f}s")
    print(sep)
    print(f"\n✓ Resultado guardado en: {CL.RESULTS_JSON}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backtest Señal Compuesta BTC/USDT")
    parser.add_argument("--fast",    action="store_true",
                        help="Usa cache .cache_compuesto/ si existe")
    parser.add_argument("--nocache", action="store_true",
                        help="Borra cache y recalcula todo desde cero")
    args = parser.parse_args()
    main(use_cache=args.fast, clear_cache=args.nocache)
